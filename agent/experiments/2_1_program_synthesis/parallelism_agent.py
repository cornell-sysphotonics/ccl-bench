#!/usr/bin/env python3
"""
Program-synthesis parallelism optimizer.

Instead of picking (tp, dp, pp) directly, the Claude agent iteratively
refines a Python *policy function* that maps workload parameters to
parallelism configurations.  The policy is evaluated on a suite of test
workloads, and the agent rewrites the function to improve aggregate
wall time.

Usage:
    python parallelism_agent.py [--config PATH] [--max-iterations N]
                                [--seed-policy PATH]
"""

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import anthropic
import yaml


# ── Paths ──────────────────────────────────────────────────────────────────────
AGENT_DIR       = Path(__file__).parent.parent.parent
API_KEY_FILE    = AGENT_DIR / "API_KEY"
DEFAULT_CONFIG  = Path(__file__).parent / "experiment.yaml"
DEFAULT_SEED    = Path(__file__).parent / "seed_policy.py"
_RUN_TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
POLICY_DIR      = Path(__file__).parent / f"policies_{_RUN_TIMESTAMP}"
CACHE_FILE      = Path(__file__).parent / "sim_cache.json"
RESULTS_CSV     = Path(__file__).parent / f"eval_results_{_RUN_TIMESTAMP}.csv"


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a program-synthesis agent that optimizes parallelism policies for \
distributed AI training.

Your task is to write and iteratively refine a Python function called `policy` \
that maps workload parameters to a parallelism configuration (tp, dp, pp, micro_batch) \
minimizing wall time.

The function signature is:
```python
def policy(
    model: str,
    batch_size: int,
    seq_len: int,
    dmodel: int,
    num_heads: int,
    num_kv_heads: int,
    num_stacks: int,
    precision: str,
    total_gpus: int,
    gpu_memory_gb: int,
    gpus_per_node: int,
    intra_node_bandwidth_gbps: int,
    inter_node_bandwidth_gbps: int,
) -> dict:
    # Return {"tp": int, "dp": int, "pp": int, "micro_batch": int}
```

Constraints:
- tp * dp * pp <= total_gpus
- tp must divide num_heads and num_kv_heads
- When pp == 1: micro_batch = batch_size / dp (fixed, no pipeline splitting). \
- When pp > 1: 1 <= micro_batch <= batch_size / (dp * pp). \
  Smaller micro_batch increases pipeline granularity and reduces bubble overhead.
- The config must not cause out-of-memory on the given GPU

Workflow (each iteration):
1. You receive the current policy code and its evaluation results across \
   ALL test workloads (wall time for each, or errors).
2. Analyze which workloads perform poorly and why. Identify patterns and \
   root causes (e.g. memory bottleneck, communication overhead, under-utilization).
3. Use the `submit_policy` tool to submit a revised, more advanced policy \
   function that addresses the issues found.
4. The new policy will be automatically evaluated on ALL test workloads, and \
   the results will be shown to you at the start of the next iteration.

Guidelines:
- The policy should be a general function, not a lookup table for the test \
  workloads. It should generalize to unseen workloads.
- Think about what drives optimal parallelism: model size vs memory, \
  communication patterns, batch size scaling with DP, etc.
- Each iteration you should make the policy MORE ADVANCED — incorporate \
  lessons learned from evaluation feedback.
- Scoring (in priority order): \
  1. SUCCESS RATE — maximize the fraction of workloads that run without error. \
     Fixing OOM/failures is MORE IMPORTANT than optimizing wall time. \
  2. AVERAGE WALL TIME of successful workloads — minimize this as a secondary objective.
- You MUST submit exactly one policy per iteration via `submit_policy`. \
  Do not end a turn without submitting.
"""


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _total_gpus(cfg: dict) -> int:
    env = cfg["environment"]
    return env["nodes"] * env["gpus_per_node"]


# ── Policy loading / saving ────────────────────────────────────────────────────

def load_policy_module(path: Path):
    """Dynamically load a policy module from a .py file."""
    spec = importlib.util.spec_from_file_location("policy_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_policy(code: str, iteration: int) -> Path:
    """Save policy code to a versioned file and return the path."""
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    path = POLICY_DIR / f"policy_v{iteration}.py"
    path.write_text(code)
    return path


# ── Simulation ─────────────────────────────────────────────────────────────────

def _update_network_yml(network_yml: Path, tp: int, dp: int, pp: int) -> None:
    """Update network.yml topology to match the parallelism config."""
    text = network_yml.read_text()
    n_gpus = tp * dp * pp
    if n_gpus <= 4:
        topology_str  = "[ FullyConnected ]"
        npus_str      = f"[ {n_gpus} ]"
        bandwidth_str = "[ 300 ]"
        latency_str   = "[ 500.0 ]"
    else:
        n_nodes       = n_gpus // 4
        topology_str  = "[ FullyConnected, Switch ]"
        npus_str      = f"[ 4, {n_nodes} ]"
        bandwidth_str = "[ 100, 25 ]"
        latency_str   = "[ 500.0, 1000.0 ]"

    text = re.sub(r"topology:\s*\[.*?\]",   f"topology: {topology_str}",   text)
    text = re.sub(r"npus_count:\s*\[.*?\]", f"npus_count: {npus_str}",     text)
    text = re.sub(r"bandwidth:\s*\[.*?\]",  f"bandwidth: {bandwidth_str}", text)
    text = re.sub(r"latency:\s*\[.*?\]",    f"latency: {latency_str}",     text)
    network_yml.write_text(text)


def _parse_metrics(text: str) -> dict:
    """Parse ASTRA-sim output into metrics."""
    if text.strip().startswith("out of memory"):
        # Try to extract memory numbers from the OOM output
        oom_details = {}
        if m := re.search(r"required[:\s]*([\d.]+)\s*(GB|MB|GiB|MiB)", text, re.IGNORECASE):
            oom_details["required"] = f"{m.group(1)} {m.group(2)}"
        if m := re.search(r"available[:\s]*([\d.]+)\s*(GB|MB|GiB|MiB)", text, re.IGNORECASE):
            oom_details["available"] = f"{m.group(1)} {m.group(2)}"
        if m := re.search(r"per.gpu[:\s]*([\d.]+)\s*(GB|MB|GiB|MiB)", text, re.IGNORECASE):
            oom_details["per_gpu"] = f"{m.group(1)} {m.group(2)}"
        # Capture any numeric memory info from the raw line
        oom_line = text.strip().splitlines()[0] if text.strip() else ""
        msg = (
            "Per-GPU memory exceeded hardware limit. "
            "Try higher TP to shard across more GPUs."
        )
        if oom_details:
            msg += f" Memory details: {oom_details}"
        elif oom_line and oom_line != "out of memory":
            msg += f" Raw: {oom_line}"
        return {
            "error": "out of memory",
            "message": msg,
            "oom_output": text.strip()[:500],
        }

    wall_times, gpu_times, comm_times = [], [], []
    for line in text.splitlines():
        if m := re.search(r"sys\[\d+\], Wall time: (\d+)", line):
            wall_times.append(int(m.group(1)))
        elif m := re.search(r"sys\[\d+\], GPU time: (\d+)", line):
            gpu_times.append(int(m.group(1)))
        elif m := re.search(r"sys\[\d+\], Comm time: (\d+)", line):
            comm_times.append(int(m.group(1)))

    if not wall_times:
        return {"error": "no metrics in output", "tail": text[-1500:]}

    avg = lambda xs: sum(xs) / len(xs)
    return {
        "wall_time":  max(wall_times),
        "gpu_time":   avg(gpu_times),
        "comm_time":  avg(comm_times),
        "num_ranks":  len(wall_times),
    }


# ── Simulation cache (persisted to disk) ──────────────────────────────────────

def _load_cache() -> dict[str, dict]:
    """Load simulation cache from disk."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict[str, dict]) -> None:
    """Save simulation cache to disk."""
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


_sim_cache: dict[str, dict] = _load_cache()


def _cache_key(workload: dict, tp: int, dp: int, pp: int,
               micro_batch: int | None) -> str:
    """Build a string cache key from workload + parallelism params."""
    return json.dumps([
        workload["model"], workload["batch_size"], workload["seq_len"],
        workload["dmodel"], workload["num_heads"], workload["num_kv_heads"],
        workload["num_stacks"], tp, dp, pp, micro_batch,
    ], separators=(",", ":"))


def run_simulation(tp: int, dp: int, pp: int, workload: dict, cfg: dict,
                   micro_batch: int | None = None) -> dict:
    """Run ASTRA-sim with the given parallelism and workload settings.
    Results are cached to disk so identical runs return instantly across runs."""
    key = _cache_key(workload, tp, dp, pp, micro_batch)
    if key in _sim_cache:
        return {**_sim_cache[key], "cached": True}

    total = _total_gpus(cfg)
    if tp * dp * pp > total:
        return {
            "error": "invalid config",
            "message": f"tp*dp*pp={tp*dp*pp} exceeds total_gpus={total}.",
        }

    workload_name = cfg["simulation"]["workload_name"]
    env           = cfg["environment"]
    example_dir   = AGENT_DIR / "tools/astra-sim-hybrid-parallelism/examples" / workload_name
    network_yml   = example_dir / "network.yml"

    _update_network_yml(network_yml, tp, dp, pp)

    workload_env = {
        "BATCH":          str(workload["batch_size"]),
        "SEQ":            str(workload["seq_len"]),
        "DMODEL":         str(workload["dmodel"]),
        "DFF":            str(workload["dff"]),
        "DVOCAL":         str(workload["dvocal"]),
        "HEAD":           str(workload["num_heads"]),
        "KVHEAD":         str(workload["num_kv_heads"]),
        "NUM_STACKS":     str(workload["num_stacks"]),
        "WEIGHT_SHARDED": str(workload["weight_sharded"]),
        "PRECISION":      str(workload.get("precision", "fp32")),
        "CHAKRA_VERSION": str(workload["chakra_schema_version"]),
    }

    env_flags = []
    for k, v in workload_env.items():
        env_flags += ["-e", f"{k}={v}"]

    cmd = [
        "docker", "run", "--rm",
        "--shm-size=8g",
        "-v", f"{AGENT_DIR}:/agent",
        "-w", f"/agent/tools/astra-sim-hybrid-parallelism/examples/{workload_name}",
        *env_flags,
        "astra-sim:latest",
        "bash", f"/agent/tools/astra-sim-hybrid-parallelism/examples/{workload_name}/run.sh",
        "-t", str(tp), "-d", str(dp), "-p", str(pp),
        "-m", str(env["gpu_memory_gb"]),
    ]
    if micro_batch is not None:
        cmd += ["-b", str(micro_batch)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=100)
    except subprocess.TimeoutExpired:
        metrics = {
            "error": (
                f"simulation timed out after 100s "
                f"(tp={tp}, dp={dp}, pp={pp}, micro_batch={micro_batch}). "
                f"Small micro_batch with large pp causes excessive pipeline steps. "
                f"Try larger micro_batch or smaller pp."
            ),
        }
        _sim_cache[key] = metrics
        _save_cache(_sim_cache)
        return metrics
    except FileNotFoundError:
        return {"error": "docker not found — is Docker running?"}

    output_file = example_dir / "output.txt"
    raw = output_file.read_text() if output_file.exists() else (proc.stdout + proc.stderr)

    metrics = _parse_metrics(raw)
    if "error" in metrics and proc.returncode != 0:
        metrics["stderr"] = proc.stderr[-1000:]
    _sim_cache[key] = metrics
    _save_cache(_sim_cache)
    return metrics


# ── Policy evaluation ──────────────────────────────────────────────────────────

def evaluate_policy(policy_path: Path, cfg: dict) -> list[dict]:
    """Evaluate a policy on all test workloads. Returns per-workload results."""
    mod = load_policy_module(policy_path)
    total = _total_gpus(cfg)
    env = cfg["environment"]
    results = []

    for wl in cfg["test_workloads"]:
        try:
            choice = mod.policy(
                model=wl["model"],
                batch_size=wl["batch_size"],
                seq_len=wl["seq_len"],
                dmodel=wl["dmodel"],
                num_heads=wl["num_heads"],
                num_kv_heads=wl["num_kv_heads"],
                num_stacks=wl["num_stacks"],
                precision=wl.get("precision", "fp32"),
                total_gpus=total,
                gpu_memory_gb=env["gpu_memory_gb"],
                gpus_per_node=env["gpus_per_node"],
                intra_node_bandwidth_gbps=env["intra_node_bandwidth_gbps"],
                inter_node_bandwidth_gbps=env["inter_node_bandwidth_gbps"],
            )
        except Exception as e:
            results.append({
                "workload": wl["name"],
                "error": f"policy raised exception: {e}",
            })
            continue

        tp, dp, pp = choice["tp"], choice["dp"], choice["pp"]
        micro_batch = choice.get("micro_batch")

        # Validate / fix micro_batch
        if dp > 0:
            if pp == 1:
                # No pipeline: micro_batch is fixed at batch_size / dp
                micro_batch = wl["batch_size"] // dp
            elif micro_batch is not None:
                max_mb = wl["batch_size"] // (dp * pp)
                if max_mb < 1:
                    max_mb = 1
                if micro_batch < 1 or micro_batch > max_mb:
                    results.append({
                        "workload": wl["name"],
                        "tp": tp, "dp": dp, "pp": pp, "micro_batch": micro_batch,
                        "error": (
                            f"micro_batch={micro_batch} out of range [1, {max_mb}] "
                            f"(batch_size={wl['batch_size']}, dp={dp}, pp={pp})."
                        ),
                    })
                    continue

        mb_str = f", mb={micro_batch}" if micro_batch is not None else ""
        print(f"  [{wl['name']}] policy → tp={tp}, dp={dp}, pp={pp}{mb_str}")

        metrics = run_simulation(tp, dp, pp, wl, cfg, micro_batch=micro_batch)
        cached = metrics.pop("cached", False)
        entry = {"workload": wl["name"], "tp": tp, "dp": dp, "pp": pp}
        if micro_batch is not None:
            entry["micro_batch"] = micro_batch
        entry.update(metrics)
        results.append(entry)
        if "wall_time" in metrics:
            print(f"    wall_time={metrics['wall_time']:,}{'  [cached]' if cached else ''}")
        else:
            print(f"    {metrics.get('error', 'unknown error')}")

    return results


def _eval_score(results: list[dict]) -> tuple[float, float]:
    """Score a policy: (success_rate, avg_wall_time_of_successes).

    Higher success_rate is better; lower avg wall time is better.
    """
    if not results:
        return (0.0, float("inf"))
    successes = [r["wall_time"] for r in results if "wall_time" in r]
    success_rate = len(successes) / len(results)
    avg_wall = sum(successes) / len(successes) if successes else float("inf")
    return (success_rate, avg_wall)


def _score_is_better(new: tuple[float, float], old: tuple[float, float]) -> bool:
    """Compare two (success_rate, avg_wall) scores. Prioritize success rate."""
    new_sr, new_avg = new
    old_sr, old_avg = old
    if new_sr > old_sr:
        return True
    if new_sr == old_sr and new_avg < old_avg:
        return True
    return False


def _format_eval_results(results: list[dict]) -> str:
    """Format evaluation results as a readable table for the LLM."""
    lines = ["Evaluation results:"]
    lines.append(f"{'Workload':<25} {'tp':>3} {'dp':>3} {'pp':>3} {'mb':>4}  {'Wall Time':>15}  Notes")
    lines.append("-" * 80)
    total_wall = 0
    n_success = 0
    n_total = 0
    for r in results:
        name = r["workload"]
        tp = r.get("tp", "?")
        dp = r.get("dp", "?")
        pp = r.get("pp", "?")
        mb = r.get("micro_batch", "-")
        if "wall_time" in r:
            wt = f"{r['wall_time']:>15,}"
            total_wall += r["wall_time"]
            n_success += 1
            n_total += 1
            note = ""
        elif "error" in r:
            wt = "FAILED".rjust(15)
            n_total += 1
            note = r["error"]
        else:
            wt = "N/A".rjust(15)
            note = ""
        lines.append(f"{name:<25} {tp:>3} {dp:>3} {pp:>3} {mb:>4}  {wt}  {note}")

    lines.append("-" * 80)
    sr = n_success / n_total * 100 if n_total else 0
    avg_wall = total_wall / n_success if n_success else 0
    lines.append(f"Success rate: {n_success}/{n_total} ({sr:.0f}%)")
    lines.append(f"Avg wall time (successes only): {avg_wall:,.0f}")
    return "\n".join(lines)


# ── CSV logging ───────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "iteration", "policy_version", "workload", "tp", "dp", "pp",
    "micro_batch", "wall_time", "gpu_time", "comm_time", "error",
    "total_wall_time", "best_total_wall_time",
]


def _init_csv() -> None:
    """Write CSV header (overwrites any existing file)."""
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)


def _append_csv(iteration: int, policy_version: int,
                eval_results: list[dict],
                total_wall: float, best_total_wall: float) -> None:
    """Append one row per workload for this iteration."""
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for r in eval_results:
            writer.writerow([
                iteration,
                policy_version,
                r.get("workload", ""),
                r.get("tp", ""),
                r.get("dp", ""),
                r.get("pp", ""),
                r.get("micro_batch", ""),
                r.get("wall_time", ""),
                r.get("gpu_time", ""),
                r.get("comm_time", ""),
                r.get("error", ""),
                total_wall,
                best_total_wall,
            ])


# ── Agent loop ─────────────────────────────────────────────────────────────────

def _build_env_description(cfg: dict) -> str:
    """Build a static description of the environment and test workloads."""
    total = _total_gpus(cfg)
    env = cfg["environment"]

    workloads_desc = []
    for wl in cfg["test_workloads"]:
        workloads_desc.append(
            f"  - {wl['name']}: model={wl['model']}, batch={wl['batch_size']}, "
            f"seq={wl['seq_len']}, dmodel={wl['dmodel']}, heads={wl['num_heads']}, "
            f"precision={wl.get('precision', 'fp32')}"
        )

    return (
        f"## Environment\n"
        f"- {total} GPUs ({env['nodes']} nodes × {env['gpus_per_node']} GPUs/node)\n"
        f"- GPU: {env['gpu_model']} {env['gpu_memory_gb']} GB\n"
        f"- Intra-node BW: {env['intra_node_bandwidth_gbps']} Gbps, "
        f"Inter-node BW: {env['inter_node_bandwidth_gbps']} Gbps\n\n"
        f"## Test Workloads\n" + "\n".join(workloads_desc)
    )


def _build_iteration_message(
    policy_code: str,
    eval_results: list[dict],
    iteration: int,
    max_iterations: int,
    best_score: tuple[float, float],
) -> str:
    """Build the per-iteration message showing current policy and results."""
    cur_sr, cur_avg = _eval_score(eval_results)
    best_sr, best_avg = best_score
    return (
        f"## Iteration {iteration}/{max_iterations}\n\n"
        f"## Current Policy\n```python\n{policy_code}\n```\n\n"
        f"## Evaluation Results (all test workloads)\n"
        f"```\n{_format_eval_results(eval_results)}\n```\n\n"
        f"Current score:  success_rate={cur_sr:.0%}, avg_wall_time={cur_avg:,.0f}\n"
        f"Best score:     success_rate={best_sr:.0%}, avg_wall_time={best_avg:,.0f}\n\n"
        f"**Your task this iteration:**\n"
        f"1. Analyze the results — which workloads FAILED and why? Fix those FIRST.\n"
        f"2. Then optimize wall time for successful workloads.\n"
        f"3. Submit an improved policy via `submit_policy`.\n"
    )


def run_agent(cfg: dict, seed_policy_path: Path, max_iterations: int = 10,
              patience: int = 5) -> Path:
    """
    Run the program-synthesis agent loop.

    Each iteration:
      1. Present current policy + full evaluation results to the agent
      2. Agent submits an improved policy via submit_policy
      3. New policy is evaluated on all workloads → next iteration

    Returns the path to the best policy file.
    """
    api_key = API_KEY_FILE.read_text().strip()
    client  = anthropic.Anthropic(api_key=api_key)
    max_tool_calls_per_iter = 5  # safety limit per iteration

    tools = [
        {
            "name": "submit_policy",
            "description": (
                "Submit a revised policy function. This ends the current iteration. "
                "The function will be evaluated on all test workloads and results "
                "shown at the start of the next iteration. The code must define "
                "a function called `policy` with the exact signature shown."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Complete Python source code defining the policy function. "
                            "Must contain a function called `policy` with the correct signature."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    ]

    # ── Evaluate seed policy ──────────────────────────────────────────────
    current_policy_path = seed_policy_path
    current_code = seed_policy_path.read_text()
    print("\n[eval] Evaluating seed policy...")
    eval_results = evaluate_policy(current_policy_path, cfg)

    best_score = _eval_score(eval_results)
    best_policy_path = current_policy_path
    policy_version = 0
    no_improve_count = 0  # early stopping counter

    # ── CSV init ──────────────────────────────────────────────────────────
    _init_csv()
    _append_csv(0, 0, eval_results, best_score[1], best_score[1])

    print(f"\n[agent] Seed policy: success_rate={best_score[0]:.0%}, avg_wall={best_score[1]:,.0f}")
    print(f"[agent] Starting refinement loop (max {max_iterations} iterations, "
          f"patience={patience})\n")

    env_desc = _build_env_description(cfg)

    # ── Main iteration loop ───────────────────────────────────────────────
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"{'='*60}")

        # Build fresh messages for this iteration (no cross-iteration history)
        iter_message = (
            env_desc + "\n\n"
            + _build_iteration_message(
                current_code, eval_results,
                iteration, max_iterations, best_score,
            )
        )
        messages = [{"role": "user", "content": iter_message}]

        # ── Inner loop: agent explores and eventually submits ─────────
        policy_submitted = False
        tool_calls = 0

        while not policy_submitted and tool_calls < max_tool_calls_per_iter:
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                tools=tools,
                tool_choice={"type": "any"},
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            # Print text blocks
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"\n[Claude] {block.text}")

            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            if tool_use is None:
                break

            tool_calls += 1

            if tool_use.name == "submit_policy":
                code = tool_use.input.get("code")
                if not code:
                    result = {"error": "submit_policy called without 'code' field. You must provide the full policy source code in the 'code' parameter."}
                    print(f"    → missing 'code' in tool input: {list(tool_use.input.keys())}")
                    messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_use.id,
                                     "content": json.dumps(result)}],
                    })
                    continue
                policy_version += 1
                policy_path = save_policy(code, policy_version)
                print(f"\n  [policy v{policy_version}] Saved to {policy_path}")

                # Validate the policy compiles
                try:
                    mod = load_policy_module(policy_path)
                    if not hasattr(mod, "policy") or not callable(mod.policy):
                        raise AttributeError("No callable 'policy' function found")
                except Exception as e:
                    result = {"error": f"Policy failed to load: {e}"}
                    print(f"    → {result}")
                    messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_use.id,
                                     "content": json.dumps(result)}],
                    })
                    # Let the agent try again within this iteration
                    continue

                # Evaluate the new policy on ALL workloads
                print(f"  [eval] Evaluating policy v{policy_version} on all workloads...")
                eval_results = evaluate_policy(policy_path, cfg)
                cur_score = _eval_score(eval_results)
                improved = _score_is_better(cur_score, best_score)

                if improved:
                    best_score = cur_score
                    best_policy_path = policy_path
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Log to CSV
                _append_csv(iteration, policy_version, eval_results,
                            cur_score[1], best_score[1])

                current_code = code
                current_policy_path = policy_path
                policy_submitted = True

                cur_sr, cur_avg = cur_score
                best_sr, best_avg = best_score
                status = "IMPROVED" if improved else "no improvement"
                print(f"    success_rate={cur_sr:.0%}, avg_wall={cur_avg:,.0f} ({status})")
                print(f"    best: success_rate={best_sr:.0%}, avg_wall={best_avg:,.0f}")

                # Send confirmation back (closes the tool call)
                result_text = _format_eval_results(eval_results)
                result_text += f"\n\nCurrent:  success_rate={cur_sr:.0%}, avg_wall_time={cur_avg:,.0f}"
                result_text += f"\nBest:     success_rate={best_sr:.0%}, avg_wall_time={best_avg:,.0f}"
                result_text += f"\n{status.upper()}"
                messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_use.id,
                                 "content": result_text}],
                })

        if not policy_submitted:
            print(f"\n[agent] Iteration {iteration}: no policy submitted, ending.")
            break

        if no_improve_count >= patience:
            print(f"\n[agent] Early stopping: no improvement for {patience} consecutive iterations.")
            break

    print(f"\n[agent] Results saved to {RESULTS_CSV}")
    return best_policy_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Program-synthesis parallelism optimizer")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG),
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--seed-policy", default=str(DEFAULT_SEED),
        help="Path to the seed policy .py file",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=15,
        help="Max policy-refinement iterations (default: 15)",
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Stop early after N iterations with no improvement (default: 3)",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    env = cfg["environment"]
    total = _total_gpus(cfg)

    print("=" * 60)
    print("Program-Synthesis Parallelism Optimizer")
    print("=" * 60)
    print(f"Environment: {total} GPUs ({env['nodes']}n × {env['gpus_per_node']}g), "
          f"{env['gpu_model']} {env['gpu_memory_gb']} GB")
    print(f"Test workloads: {len(cfg['test_workloads'])}")
    print(f"Seed policy: {args.seed_policy}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Patience: {args.patience}")
    print("=" * 60)

    best_path = run_agent(
        cfg,
        seed_policy_path=Path(args.seed_policy),
        max_iterations=args.max_iterations,
        patience=args.patience,
    )

    print("\n" + "=" * 60)
    print(f"Best policy: {best_path}")
    print("=" * 60)
    print("\nFinal policy code:")
    print(best_path.read_text())


if __name__ == "__main__":
    main()
