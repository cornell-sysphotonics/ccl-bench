"""
update_policy step — CCL-Search loop, step 4.

Given the execution history and the current generate_config program, asks the
LLM to produce an improved version. The LLM sees all past run records (configs
tried, metrics measured, scores) and submits a revised generate_config via the
`submit_config` tool.

Interface:
    gc_path, code = update_policy(context) or (None, None) if no submission.

    update_policy(
        gc_code, gc_dir, version,
        history, workload, environment, tuning,
        client, iteration, max_iterations, best_score,
    ) -> tuple[Path, str] | tuple[None, None]
"""

import importlib.util
import json
from pathlib import Path

import anthropic


SYSTEM_PROMPT = """\
You are CCL-Search, a configuration optimization agent for LLM infrastructure built on CCL-Bench.

Your goal: write and iteratively refine a Python function `generate_config` that maps
workload cards and environment descriptors to configuration key-value pairs optimizing
the user-defined performance objective.

## Function signature

```python
def generate_config(workload: dict, environment: dict) -> dict:
    \"""
    Args:
        workload:    Workload card fields — model_family, phase, batch_size, seq_len,
                     num_heads, num_layers, precision, moe (bool), config_space (list
                     of tunable dimensions with valid choices), run_script, trace_dir.
        environment: Hardware/software descriptor — gpu_model, gpu_memory_gb,
                     total_gpus, gpus_per_node, intra/inter_node_bandwidth_gbps,
                     framework, framework_version.

    Returns:
        dict of configuration key-value pairs matching config_space keys, e.g.:
          {"tp": 4, "dp": 8, "pp": 1, "micro_batch_size": 4, "activation_checkpointing": True}
    \"""
```

## Context you receive

- The current `generate_config` source policy.
- Full execution history: every config tried, its measured metrics, and its score.
  Use this to understand what worked, what failed, and why.
- A summary table of all runs with scores.

## Workflow

1. Analyse the history — which configs performed best, which failed and why.
2. Submit an improved `generate_config` via `submit_config`.
3. The new function is executed immediately; results appear next iteration.

## Scoring

score = weighted sum of CCL-Bench metrics (lower is better when `minimize`).
Priority 1 — fix errors/timeouts. Priority 2 — improve the score.

## Design guidance for adaptive policies

Write general, model-aware, environment-aware logic — NOT a fixed config dict. The policy should reason
about the workload and hardware to pick the right parallelism strategy:

### Key workload fields to use:
- `workload["model_family"]` — model name (e.g., "llama-3.1-8b", "deepseek-v2-lite")
- `workload["moe"]` — True if Mixture-of-Experts model
- `workload["num_layers"]` — number of transformer layers (affects PP choices)
- `workload["num_params"]` — total parameter count (affects memory requirements)
- `workload["batch_size"]` — global batch size
- `workload["seq_len"]` — sequence length
- `workload["config_space"]` — list of tunable dimensions with valid choices

### Key environment fields to use:
- `environment["gpu_memory_gb"]` — GPU memory (e.g., 40 for A100-40GB)
- `environment["total_gpus"]` — total GPUs available
- `environment["gpus_per_node"]` — GPUs per node (e.g., 4)
- `environment["inter_node_bandwidth_gbps"]` — inter-node bandwidth (affects TP/DP/EP tradeoffs)

### Parallelism reasoning principles:
1. **TP (tensor parallelism):** Keep TP within a node (tp <= gpus_per_node). Higher TP
   reduces per-GPU memory but adds allreduce communication. On slow interconnects,
   minimize cross-node TP.
2. **PP (pipeline parallelism):** PP must divide num_layers evenly. PP adds pipeline
   bubble overhead proportional to (PP-1)/num_microbatches. Use PP to reduce memory
   when TP alone isn't enough.
3. **DP (data parallelism):** Scales throughput but adds allreduce for gradients.
   On slow interconnects, DP across nodes is expensive.
4. **EP (expert parallelism, MoE only):** EP must divide num_experts and dp.
   Distributes experts across GPUs. More EP = less per-GPU expert memory but more
   alltoall communication for token routing. On slow interconnects, alltoall is
   very expensive.
5. **Memory estimation:** Total training memory per GPU ≈
   (num_params × 12 bytes) / (tp × pp) + activation_memory.
   Must fit in gpu_memory_gb. Use activation_checkpointing=True if tight.
6. **micro_batch_size:** Larger = better GPU utilization but more activation memory.
   Start with 1, try 2, then 4. If OOM, reduce.
7. **tp × dp × pp** does NOT need to equal total_gpus. Using fewer GPUs can be
   more efficient if the model fits.

### MoE-specific guidance:
- MoE models have sparse experts — only a subset are active per token
- EP distributes experts across GPUs, reducing memory but adding alltoall communication
- On slow interconnects (< 100 Gbps), alltoall for expert routing dominates step time
- EP should divide both num_experts and dp
- With EP, allgather/reducescatter for FSDP parameter sync can dominate — check if
  reducing DP or increasing EP helps

### Example adaptive policy structure:
```python
def generate_config(workload, environment):
    gpu_mem = environment.get("gpu_memory_gb", 40)
    gpus_per_node = environment.get("gpus_per_node", 4)
    total_gpus = environment.get("total_gpus", 16)
    is_moe = workload.get("moe", False)
    num_layers = workload.get("num_layers", 32)
    
    # Parse config space for valid choices
    config_space = {d["key"]: d["choices"] for d in workload.get("config_space", [])}
    
    # Start with TP fitting within a node
    tp = min(gpus_per_node, max(config_space.get("tp", [1])))
    
    # Estimate memory and adjust parallelism
    # ... model-specific logic ...
    
    return {"tp": tp, "dp": dp, "pp": pp, ...}
```

## Important constraints
- Each entry in config_space is a dict: `{"key": "tp", "type": "int", "choices": [1,2,4], "description": "..."}`.
  Always use `dim["key"]` (not `dim["name"]`) to get the dimension name.
- Each iteration should incorporate lessons learned from the history.
- You MUST call `submit_config` exactly once per iteration.
"""


# ── Formatting ─────────────────────────────────────────────────────────────────

def _format_history_table(history: list[dict], optimization_goal: dict) -> str:
    direction    = optimization_goal.get("direction", "minimize")
    metric_names = [mc["name"] for mc in optimization_goal.get("metrics", [])]
    hdr  = "  ".join(f"{n[:12]:>12}" for n in metric_names)
    sep  = "-" * 85
    lines = [f"{'#':>3}  {'Config':<36}  {hdr}  {'Score':>12}  Status", sep]
    for i, r in enumerate(history):
        cfg = ", ".join(f"{k}={v}" for k, v in sorted(r.get("config", {}).items()))[:36]
        if r.get("status") == "success":
            mv = "  ".join(f"{r['metrics'].get(n, float('nan')):>12.4g}"
                           for n in metric_names)
            sc = f"{r['score']:>12.4g}"
            st = "ok"
        else:
            mv = "  ".join(f"{'—':>12}" for _ in metric_names)
            sc = f"{'FAILED':>12}"
            st = f"{r.get('status','?')}: {(r.get('error_msg') or '')[:30]}"
        lines.append(f"{i+1:>3}  {cfg:<36}  {mv}  {sc}  {st}")
    ok   = [r for r in history if r.get("status") == "success"]
    best = min((r["score"] for r in ok), default=float("inf"))
    lines += [sep,
              f"Runs: {len(history)}  |  OK: {len(ok)}  |  "
              f"Best score: {best:.4g}  [{direction}]"]
    return "\n".join(lines)


def _build_message(
    gc_code: str,
    history: list[dict],
    iteration: int,
    max_iterations: int,
    best_score: float,
    workload: dict,
    environment: dict,
    tuning: dict,
) -> str:
    goal    = tuning["optimization_goal"]
    env_str = "\n".join(f"  {k}: {v}" for k, v in environment.items() if v is not None)
    cs_str  = "\n".join(
        f"  {c['key']} ({c['type']}) choices={c.get('choices','?')} — {c.get('description','')}"
        for c in workload.get("config_space", [])
    )
    goal_str = (f"  direction: {goal.get('direction','minimize')}\n" +
                "\n".join(f"  - {mc['name']} (weight={mc.get('weight',1.0)})"
                          for mc in goal.get("metrics", [])))
    recent_json = json.dumps(history[-20:], indent=2, default=str)
    if len(recent_json) > 5000:
        recent_json = recent_json[:5000] + "\n... (truncated)"

    return (
        f"## Iteration {iteration}/{max_iterations}\n\n"
        f"## Workload\n"
        f"  {workload.get('model_family','?')} [{workload.get('phase','?')}]  "
        f"batch={workload.get('batch_size','?')}  seq={workload.get('seq_len','?')}  "
        f"precision={workload.get('precision','?')}  "
        f"moe={workload.get('moe', False)}  "
        f"num_layers={workload.get('num_layers','?')}  "
        f"num_params={workload.get('num_params','?')}\n\n"
        f"## Environment\n{env_str}\n\n"
        f"## Config Space\n"
        f"(Each entry is a dict with fields `key`, `type`, `choices`, `description`.\n"
        f"Access the dimension name as `dim['key']`, not `dim['name']`.)\n"
        f"{cs_str}\n\n"
        f"## Optimization Objective\n{goal_str}\n\n"
        f"## Current generate_config\n```python\n{gc_code}\n```\n\n"
        f"## Execution History\n"
        f"```\n{_format_history_table(history, goal)}\n```\n\n"
        f"```json\n{recent_json}\n```\n\n"
        f"Best so far:  score={best_score:.4g}\n\n"
        f"**Task:** fix failures first, then improve the score. "
        f"Write an adaptive generate_config that reasons about the workload "
        f"and hardware — not just a fixed config dict. "
        f"Submit via `submit_config`.\n"
    )


# ── gc file helpers ────────────────────────────────────────────────────────────

def _save_gc(code: str, gc_dir: Path, version: int) -> Path:
    gc_dir.mkdir(parents=True, exist_ok=True)
    path = gc_dir / f"generate_config_v{version}.py"
    path.write_text(code)
    return path


def _load_gc(path: Path):
    spec = importlib.util.spec_from_file_location("gc_module", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── update_policy ──────────────────────────────────────────────────────────────

def update_policy(
    gc_code: str,
    gc_dir: Path,
    version: int,
    history: list[dict],
    workload: dict,
    environment: dict,
    tuning: dict,
    client: anthropic.Anthropic,
    iteration: int,
    max_iterations: int,
    best_score: float,
) -> tuple[Path, str] | tuple[None, None]:
    """Ask the LLM to produce an improved generate_config program.

    Returns (gc_path, code) on success, (None, None) if the agent did not submit.
    """
    submit_tool = {
        "name": "submit_config",
        "description": (
            "Submit a revised `generate_config` function. It will be executed "
            "immediately; results appear next iteration. "
            "Must define `generate_config(workload, environment) -> dict`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python module defining `generate_config(workload, environment) -> dict`.",
                },
                "rationale": {
                    "type": "string",
                    "description": "What changed and why.",
                },
            },
            "required": ["code"],
        },
    }

    messages = [{"role": "user", "content": _build_message(
        gc_code, history, iteration, max_iterations,
        best_score, workload, environment, tuning,
    )}]

    for _ in range(5):
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            tools=[submit_tool],
            tool_choice={"type": "any"},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[Claude] {block.text}")

        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            break

        code      = tool_use.input.get("code", "")
        rationale = tool_use.input.get("rationale", "")
        if rationale:
            print(f"\n  [rationale] {rationale}")

        if not code:
            messages.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": tool_use.id,
                "content": json.dumps({"error": "'code' is required."}),
            }]})
            continue

        gc_path = _save_gc(code, gc_dir, version)
        print(f"\n  [v{version}] {gc_path.name}")

        try:
            mod = _load_gc(gc_path)
            if not callable(getattr(mod, "generate_config", None)):
                raise AttributeError("no callable 'generate_config'")
        except Exception as e:
            messages.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": tool_use.id,
                "content": json.dumps({"error": f"load failed: {e}"}),
            }]})
            continue

        # Acknowledge so the conversation closes cleanly
        messages.append({"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": tool_use.id,
            "content": "Accepted. Executing now.",
        }]})
        return gc_path, code

    return None, None
