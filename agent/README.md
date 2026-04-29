# CCL-Bench Agent

An ADRS (Automated Distributed Runtime Search) loop that uses Claude to
iteratively discover the best parallelism and compiler configuration for a
given workload.  Each iteration runs a real training/inference job, measures
a CCL-Bench metric from the collected traces, and asks Claude to improve
`generate_config` based on the results.

---

## How it works

```
┌─────────────────────────────────────────────────────────────────┐
│                       ADRS loop (agent.py)                      │
│                                                                 │
│  seed generate_config.py                                        │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. generate_config(workload, env) → config dict          │  │
│  │ 2. execute(workload, config)      → RunResult            │  │
│  │    └─ bash run_script  (env vars injected from config)   │  │
│  │       └─ writes profiler traces to TRACE_DIR             │  │
│  │ 3. compute_metric(RunResult, goal) → score               │  │
│  │    └─ tools/main.py --trace TRACE_DIR --metric <name>    │  │
│  │ 4. update_policy(history → Claude) → new generate_config │  │
│  │ 5. update_history → appended to workload card YAML       │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │ repeat up to --max-iterations                         │
└─────────────────────────────────────────────────────────────────┘
```

### Key concepts

| Term | What it is |
|---|---|
| **workload card** | YAML describing the model, hardware, and run script |
| **tuning config** | YAML defining the search space and optimization goal |
| **generate_config** | Python module the LLM refines each iteration; maps `(workload, environment)→config dict` |
| **run script** | Bash script invoked by `execute.py`; receives config choices as uppercase env vars |
| **TRACE_DIR** | Directory where the run script deposits profiler traces consumed by the metric tools |

---

## Adapting to a new environment

### Step 1 — Create an experiment folder

Each deployment lives in its own folder under `agent/`.  Use the
Perlmutter/torchtitan experiment as a reference:

```
agent/
  ccl_bench_agent/          ← shared agent code (do not modify)
  perlmutter_torchtitan_llama8b/   ← reference experiment
  your_experiment/          ← your new experiment
    workload_card.yaml
    tuning_config.yaml
    generate_config.py
    scripts/
      run.sh
```

### Step 2 — Write `workload_card.yaml`

Describes the hardware, model, and where to find the run script.

```yaml
version: 1
description: "Llama-3.1-8B on 8× A100, PyTorch FSDP"

workload:
  model:
    phase: training
    model_family: Llama-3.1-8B
    precision: bf16
    num_params: 8030261248
    num_layers: 32
    num_heads: 32
    num_kv_heads: 8
  data:
    batch_size: 32
    seq_len: 1024
    dataset: random
  hardware:
    network_topo:
      topology: nvlink
      bandwidth_gbps:
        - 600   # intra-node
    xpu_spec:
      type: GPU
      model: nvidia_a100
      memory_gb: 80
      total_count: 8
      count_per_node: 8

Model-executor:
  framework:
    name: torchrun
    version: "2.4.0"
    # Absolute path — execute.py calls `bash run_script`
    run_script: /absolute/path/to/your_experiment/scripts/run.sh
    # TRACE_DIR: where the run script writes traces
    trace_dir: /tmp/ccl_bench_traces/your_experiment/

metric_source:
  traces:
    - json   # PyTorch profiler kineto JSON
    # alternatives: json_tpu  (XLA Chrome trace)
```

**For TPUs** use `xpu_spec.type: TPU`, set `model` (e.g. `v4-8`), and set
`metric_source.traces: [json_tpu]`.

### Step 3 — Write `tuning_config.yaml`

Defines what the agent can tune and what it is optimising.

```yaml
config_space:
  # Each entry becomes an uppercase env var passed to the run script.
  - key: tp
    type: int
    description: Tensor parallelism degree
    choices: [1, 2, 4, 8]
  - key: dp
    type: int
    description: FSDP data parallelism degree
    choices: [1, 2, 4, 8]
  - key: activation_checkpointing
    type: bool
    description: Full activation checkpointing
    choices: [true, false]
  # Add or remove dimensions freely; the run script must honour them.

optimization_goal:
  direction: minimize          # or maximize
  metrics:
    - name: avg_step_time      # must match a tool in tools/
      weight: 1.0
  # Composite example:
  # metrics:
  #   - name: avg_step_time
  #     weight: 0.7
  #   - name: communication_fraction
  #     weight: 0.3
```

Available metric names come from `tools/` (one subdirectory = one metric):
`avg_step_time`, `communication_fraction`, `mfu`, `coll_call_num`, etc.

### Step 4 — Write the run script

`execute.py` calls `bash run_script` with the config dict injected as
uppercase environment variables.  It also sets:

| Env var | Value |
|---|---|
| `TRACE_DIR` | from `workload_card.yaml → trace_dir` |
| `WORKLOAD_NAME` | from `workload_card.yaml → description` |
| `<KEY>` | one uppercase var per `config_space` key (e.g. `TP=4`, `DP=8`) |

The script **must**:
1. Use the env vars to launch the training/inference job.
2. Emit profiler traces into `$TRACE_DIR` in a format the chosen metric tool
   understands (see table below).
3. Exit 0 on success, non-zero on failure.

```bash
#!/bin/bash
set -e

TP=${TP:-1}
DP=${DP:-1}
TRACE_DIR=${TRACE_DIR:-"/tmp/traces"}

# ... launch your job using $TP, $DP, etc. ...

# PyTorch profiler traces must be flat *.json files in $TRACE_DIR:
#   $TRACE_DIR/rank0_trace.json
#   $TRACE_DIR/rank1_trace.json
#   ...
# If your framework nests them, copy them up:
#   cp profiler_output/iteration_N/*.json "$TRACE_DIR/"

# avg_step_time also needs a YAML with metric_source in $TRACE_DIR:
cat > "$TRACE_DIR/trace_meta.yaml" << 'YAML'
metric_source:
  traces:
    - json
YAML
```

#### Trace format requirements per metric backend

| `metric_source.traces` value | Required file format | Typical source |
|---|---|---|
| `json` | PyTorch kineto JSON with `ProfilerStep#N` user annotations | `torch.profiler.profile` |
| `json_tpu` | XLA Chrome-trace JSON with `$core.py:331 step` events | JAX/XLA profiler |

**PyTorch profiler** — enable with `torch.profiler.profile` and a step-based
schedule so that `ProfilerStep#N` spans appear in the trace.  Export each
rank's trace as `rank{r}_trace.json` directly in `$TRACE_DIR`.

**XLA/TPU profiler** — use `jax.profiler.trace` or equivalent and point the
output directory to `$TRACE_DIR`.

### Step 5 — Write `generate_config.py` (seed policy)

The seed is the starting point for the LLM search.  It should return a
*valid* config (no errors on first run).

```python
def generate_config(workload: dict, environment: dict) -> dict:
    total_gpus    = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    # Baseline: fill one node with TP, spread DP across nodes
    tp = min(gpus_per_node, 8)
    dp = max(1, total_gpus // tp)
    return {
        "tp": tp,
        "dp": dp,
        "activation_checkpointing": False,
    }
```

The function receives the flattened workload and environment fields defined
in `agent.py:_flatten_workload` / `_flatten_environment`.  The returned
dict must have keys that match `config_space` entries.

---

## Running the agent

```bash
# From the ccl_bench_agent directory
cd agent/ccl_bench_agent

# Prerequisites
pip install anthropic pyyaml
echo "sk-ant-..." > ../API_KEY

# Launch
python agent.py \
  --card    ../your_experiment/workload_card.yaml \
  --tuning  ../your_experiment/tuning_config.yaml \
  --seed    ../your_experiment/generate_config.py \
  --max-iterations 20 \
  --patience 5
```

The agent writes two outputs alongside the agent source:

| Output | Contents |
|---|---|
| `results_<timestamp>.csv` | One row per iteration: config, metrics, score, best_score |
| `generate_config_<timestamp>/generate_config_vN.py` | Each LLM-revised policy version |

The workload card YAML is also updated in-place with a `runs:` section
containing every execution record (for upload to the CCL-Bench platform).

---

## Example: Perlmutter / torchtitan / Llama-3.1-8B

See `agent/perlmutter_torchtitan_llama8b/` for a complete working example:

```bash
cd agent/ccl_bench_agent
python agent.py \
  --card    ../perlmutter_torchtitan_llama8b/workload_card.yaml \
  --tuning  ../perlmutter_torchtitan_llama8b/tuning_config.yaml \
  --seed    ../perlmutter_torchtitan_llama8b/generate_config.py
```

The run script (`scripts/train_llama8b.sh`) handles single-node `torchrun`
and multi-node `srun + torchrun` automatically, and flattens the torchtitan
profiler traces from their nested `profile_traces/iteration_N/` structure
into the flat `$TRACE_DIR/` layout that `avg_step_time` expects.

---

## Directory reference

```
agent/
  ccl_bench_agent/
    agent.py           — ADRS loop orchestrator; entry point
    execute.py         — step 2: runs run_script, returns RunResult
    compute_metric.py  — step 3: calls tools/main.py to score traces
    update_policy.py   — step 4: asks Claude to improve generate_config
    update_history.py  — step 5: appends record to card YAML
    generate_config.py — default seed policy (template)
    workload_card.yaml — template workload card
    tuning_config.yaml — template tuning config

  perlmutter_torchtitan_llama8b/   — reference experiment
    workload_card.yaml
    tuning_config.yaml
    generate_config.py
    scripts/train_llama8b.sh

  experiments/         — simulation-based experiments (separate from agent)
  torchtitan/          — torchtitan source (framework backend)

tools/                 — CCL-Bench metric tools
  main.py              — dispatches --metric name to the right tool
  avg_step_time/
  communication_fraction/
  mfu/
  ...
```
