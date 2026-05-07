# CCL-Search

A configuration search loop that uses Claude to iteratively discover the best
parallelism and compiler configuration for a given workload.  Each iteration
runs a real training/inference job, measures a CCL-Bench metric from the
collected traces, and asks Claude to improve `generate_config` based on the
results.

---

## How it works

```
┌─────────────────────────────────────────────────────────────────┐
│                    CCL-Search loop (agent.py)                   │
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
dry-run example as a reference (no hardware required):

```
agent/
  ccl_bench_agent/          ← shared agent code (do not modify)
  ccl_bench_agent/dry_run/  ← reference example (synthetic traces, no hardware)
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
    # Must be on a shared filesystem (not /tmp) so traces are visible
    # from the node running the agent (login node or head node).
    trace_dir: /shared/path/ccl_bench_traces/your_experiment/

metric_source:
  traces:
    - json   # PyTorch profiler kineto JSON
    # alternatives: json_tpu  (XLA Chrome trace)
```

**For TPUs** use `xpu_spec.type: TPU`, set `model` (e.g. `v4-8`), and set
`metric_source.traces: [json_tpu]`.

#### Choosing `trace_dir`

`trace_dir` must be visible from wherever the agent process runs (typically
the login or head node) **and** from the compute nodes running the job.

| Filesystem | Suitable? | Notes |
|---|---|---|
| Lustre / GPFS (shared scratch) | **Yes** | Shared across all nodes; use the cluster's shared scratch filesystem |
| `/tmp` | **No** | Node-local; traces written on compute nodes are invisible on login node |
| NFS home (`$HOME`) | Sometimes | Available but may be slow for large traces |

Set `trace_dir` to a path on your cluster's shared filesystem:
```yaml
trace_dir: /shared/scratch/<username>/ccl-bench-traces/your_experiment/
```

Also update the default in your run script:
```bash
TRACE_DIR=${TRACE_DIR:-"/shared/scratch/yourname/ccl-bench-traces/your_experiment"}
```

### Step 3 — Write `tuning_config.yaml`

Defines what the agent can tune and what it is optimising.

```yaml
config_space:
  # Each entry becomes an uppercase env var passed to the run script.
  # Use `key` (not `name`) to identify each dimension.
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
TRACE_DIR=${TRACE_DIR:-"/shared/scratch/yourname/ccl-bench-traces/your_experiment"}

mkdir -p "$TRACE_DIR"

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
dict must have keys that match `config_space` entries.  Access dimension
metadata with `dim["key"]` (not `dim["name"]`):

```python
config_space = workload.get("config_space", [])
valid_tp = next(d["choices"] for d in config_space if d["key"] == "tp")
```

---

## Running the agent

```bash
# From the ccl_bench_agent directory
cd agent/ccl_bench_agent

# Prerequisites
pip install anthropic pyyaml requests
echo "sk-ant-..." > ../API_KEY

# Launch
python agent.py \
  --card        ../your_experiment/workload_card.yaml \
  --tuning      ../your_experiment/tuning_config.yaml \
  --seed        ../your_experiment/generate_config.py \
  --max-iterations 20 \
  --patience 5

# Override output directory (default: ccl_bench_agent/runs/)
python agent.py --card ... --output-dir /path/to/my_runs
```

### Output layout

Each agent run creates a timestamped subdirectory under `runs/`:

```
ccl_bench_agent/
  runs/
    20260428_202809/        ← one directory per agent invocation
      agent.log             ← full console transcript
      results.csv           ← one row per iteration
      results.png           ← plot (generated separately, see below)
      policies/
        generate_config_v1.py
        generate_config_v2.py
        ...
  run_cache.json            ← result cache shared across all runs
```

`results.csv` columns:

| Column | Description |
|---|---|
| `iteration` | Loop iteration index (0 = seed run) |
| `version` | Policy version number |
| `config` | JSON of the config dict tried |
| `metrics` | JSON of measured metric values |
| `score` | Weighted objective score |
| `status` | `success`, `error`, or `timeout` |
| `error_msg` | Error details on failure |
| `best_score` | Running best score up to this iteration |
| `search_time_s` | Seconds spent in LLM `update_policy` call |

The workload card YAML is also updated in-place with a `runs:` section
containing every execution record.

### Result cache

`run_cache.json` persists the result of every `(run_script, config)` pair.
If the agent is restarted or a duplicate config is proposed, the result is
served from cache without re-running the job.  Delete the file to force
a fresh run of all configs.

---

## Plotting results

```bash
cd agent/ccl_bench_agent

# Plot the most recent run (auto-discovers latest runs/*/results.csv):
python plot_results.py

# Plot a specific run:
python plot_results.py runs/20260428_202809/results.csv
```

The script saves `results.png` alongside the CSV with three panels:

1. **Score per iteration** — scatter (green = success, red = failed) with a
   running-best line
2. **LLM search time** — seconds spent in `update_policy` per iteration
3. **Config heatmap** — each config dimension as a row, each iteration as a
   column, annotated with values

---

## Uploading traces to CCL-Bench

After a run completes, upload traces and the populated workload card to the
CCL-Bench upload server using `upload_traces.py`.

```bash
cd agent/ccl_bench_agent

# Upload all successful runs recorded in the workload card:
python upload_traces.py \
  --server http://your-server:5000 \
  --card   ../your_experiment/workload_card.yaml

# Check what would be sent without hitting the server:
python upload_traces.py --card ... --dry-run

# Also upload failed runs:
python upload_traces.py --card ... --include-failed

# Upload a single trace directory manually:
python upload_traces.py \
  --server    http://your-server:5000 \
  --card      ../your_experiment/workload_card.yaml \
  --trace-dir /shared/scratch/yourname/ccl-bench-traces/run1 \
  --group     "tp4_dp4_manual" \
  --desc      "manual rerun"
```

Set the server URL once via environment variable to avoid repeating it:

```bash
export CCL_UPLOAD_URL=http://your-server:5000
python upload_traces.py --card ../your_experiment/workload_card.yaml
```

Each run is uploaded as a separate group named `iter{N}_{config}` (e.g.
`iter003_dp4pp1tp4`), with the full config and metrics embedded in the
description and the workload card attached as JSON.

---

## Example: Simulation (no hardware required)

`ccl_bench_agent/dry_run/` contains a complete self-contained simulation that
runs without any GPU, cluster, or network connection.  A synthetic trace
generator (`mock_trace_gen.py`) models Llama-3.1-8B step time on a 4-GPU
single-node setup using a parametric formula:

| Component | Formula | Notes |
|---|---|---|
| TP compute | `1.1 / tp` | Linear speedup |
| TP comm | `tp × 0.008` | AllReduce cost grows with group size |
| DP grad sync | `0.04 × (1 − 1/dp)` | Only when `dp > 1` |
| PP bubble | `(pp−1) × 0.08 / micro_batch` | Idle time between pipeline stages |
| Activation ckpt | `+0.10` | ~10 % re-compute overhead |

Memory constraint: OOM (exit 1) if `80 GB / (tp × pp) > 40 GB per GPU`.

Expected optimum: `tp=4, dp=1, pp=1, micro_batch=1, act_ckpt=false` → ~0.31 s.

### Prerequisites

```bash
pip install anthropic pyyaml
export ANTHROPIC_API_KEY=sk-ant-...
```

### Running

```bash
# Run from anywhere — the script cd-s into ccl_bench_agent/ automatically:
bash agent/ccl_bench_agent/dry_run/run.sh
```

### Sample session

```
=================================================================
CCL-Search: Configuration Optimization Agent
=================================================================
Card:       workload_card.yaml
Tuning:     tuning_config.yaml
Workload:   llama-3.1-8b  [training]
Hardware:   4 × generic_gpu_40gb
Objective:  minimize  avg_step_time × 1.0
Seed:       generate_config.py
Max iters:  10  patience=4
Publish to: (not set — skipping disk publish)
=================================================================

[agent] Run dir → runs/20260507_120000

[eval] Seed generate_config...
    config → activation_checkpointing=False  dp=1  micro_batch=1  pp=1  tp=1
    [mock] tp=1 dp=1 pp=1 micro_batch=1 act_ckpt=False → step_time=1.187s
    score=1.187  (avg_step_time=1.187)

[agent] Seed: status=success, score=1.187
[agent] Loop: max=10, patience=4

=================================================================
  ITERATION 1/10
=================================================================
    config → activation_checkpointing=False  dp=1  micro_batch=1  pp=1  tp=2
    [mock] tp=2 dp=1 pp=1 micro_batch=1 act_ckpt=False → step_time=0.586s
    score=0.5861  (avg_step_time=0.5861)
    IMPROVED — best score=0.5861

=================================================================
  ITERATION 2/10
=================================================================
    config → activation_checkpointing=False  dp=1  micro_batch=1  pp=1  tp=4
    [mock] tp=4 dp=1 pp=1 micro_batch=1 act_ckpt=False → step_time=0.317s
    score=0.3170  (avg_step_time=0.317)
    IMPROVED — best score=0.317

=================================================================
  ITERATION 3/10
=================================================================
    config → activation_checkpointing=True  dp=1  micro_batch=1  pp=1  tp=4
    [mock] tp=4 dp=1 pp=1 micro_batch=1 act_ckpt=True → step_time=0.419s
    score=0.4190  (avg_step_time=0.419)
    no improvement — best score=0.317

=================================================================
  ITERATION 4/10
=================================================================
    config → activation_checkpointing=False  dp=1  micro_batch=2  pp=2  tp=4
    [mock] tp=4 dp=1 pp=2 micro_batch=2 act_ckpt=False → step_time=0.364s
    score=0.3640  (avg_step_time=0.364)
    no improvement — best score=0.317

...

[agent] Early stop: 4 non-improving iterations.

[agent] Results CSV → runs/20260507_120000/results.csv
[agent] Workload card → dry_run/workload_card.yaml

Best → runs/20260507_120000/policies/generate_config_v2.py
```

The agent converges to `tp=4` within 2 iterations, then spends the remaining
patience budget confirming that adding activation checkpointing or pipeline
stages makes things worse.  No real hardware or network calls are made — every
"job" is a ~1 ms Python invocation of `mock_trace_gen.py`.

---

## Directory reference

```
agent/
  ccl_bench_agent/
    agent.py            — CCL-Search loop orchestrator; entry point
    execute.py          — step 2: runs run_script, returns RunResult
    compute_metric.py   — step 3: calls tools/main.py to score traces
    update_policy.py    — step 4: asks Claude to improve generate_config
    update_history.py   — step 5: appends record to card YAML
    run_cache.py        — result cache (keyed by run_script + config)
    plot_results.py     — plot runs/*/results.csv → results.png
    upload_traces.py    — upload traces + workload card to CCL-Bench server
    generate_config.py  — default seed policy (template)
    workload_card.yaml  — template workload card
    tuning_config.yaml  — template tuning config
    runs/               — agent output (created at runtime)
      <timestamp>/
        agent.log
        results.csv
        results.png
        policies/
          generate_config_v1.py
          ...
    run_cache.json      — persisted result cache

  ccl_bench_agent/dry_run/   — reference example (no hardware)
    mock_trace_gen.py   — synthetic trace generator
    mock_run.sh         — run script wrapper
    workload_card.yaml  — generic Llama-3.1-8B on 4-GPU
    tuning_config.yaml  — search space (tp/dp/pp/micro_batch/act_ckpt)
    run.sh              — convenience launcher

  torchtitan/           — torchtitan source (framework backend)

tools/                  — CCL-Bench metric tools
  main.py               — dispatches --metric name to the right tool
  avg_step_time/
  communication_fraction/
  mfu/
  ...
```
