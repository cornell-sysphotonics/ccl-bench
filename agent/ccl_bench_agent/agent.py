#!/usr/bin/env python3
"""
CCL-Bench ADRS Configuration Tuning Agent.

Iteratively synthesizes a Python program `generate_config` that maps workload
cards and environment descriptors to optimal configuration key-value pairs.

ADRS loop:
  1. generate_config(workload, environment, history) → config dict
  2. execute   — run workload via run_script, collect traces     [execute.py]
  3. compute_metric — CCL-Bench tool scripts → objective score  [compute_metric.py]
  4. update_policy  — LLM refines generate_config

Run records are appended to the workload card after every execution so the
populated card (with traces) can be uploaded to the CCL-Bench platform.

Usage:
    python agent.py [--card PATH] [--seed PATH]
                    [--max-iterations N] [--patience N]
"""

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anthropic
import yaml


# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent
_REPO_ROOT   = _HERE.parent.parent
TOOLS_MAIN   = _REPO_ROOT / "tools" / "main.py"
API_KEY_FILE = _HERE.parent / "API_KEY"
DEFAULT_CARD   = _HERE / "workload_card.yaml"
DEFAULT_TUNING = _HERE / "tuning_config.yaml"
DEFAULT_SEED   = _HERE / "generate_config.py"
_TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
GC_DIR       = _HERE / f"generate_config_{_TIMESTAMP}"
RESULTS_CSV  = _HERE / f"results_{_TIMESTAMP}.csv"


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a configuration optimization agent for LLM infrastructure (CCL-Bench / ADRS).

Your goal: write and iteratively refine a Python function `generate_config` that maps
workload cards and environment descriptors to configuration key-value pairs optimizing
the user-defined performance objective.

## Function signature

```python
def generate_config(
    workload: dict,
    environment: dict,
    history: list[dict],
) -> dict:
    \"""
    Args:
        workload:    Workload card fields — model_family, phase, batch_size, seq_len,
                     num_heads, num_layers, precision, config_space (list of tunable
                     dimensions with valid choices), run_script, trace_dir, etc.
        environment: Hardware/software descriptor — gpu_model, gpu_memory_gb,
                     total_gpus, gpus_per_node, intra/inter_node_bandwidth_gbps,
                     framework, framework_version.
        history:     Past execution records:
                       [{"config": {...},
                         "metrics": {"metric_name": value, ...},
                         "score": float,
                         "status": "success" | "error" | "timeout"}, ...]
                     Empty list on the first call.

    Returns:
        dict of configuration key-value pairs matching config_space keys, e.g.:
          {"tp": 4, "dp": 8, "pp": 1, "micro_batch": 4, "compile_mode": "inductor"}
    \"""
```

## Workflow

Each iteration:
1. You receive the current `generate_config` source, the full execution history,
   and a table summarising configs tried and their scores.
2. Analyse the history: which configs worked, which failed and why, what
   patterns emerge from the hardware environment.
3. Submit an improved `generate_config` via `submit_config`.
4. The new function is executed immediately; results appear next iteration.

## Scoring

Score = weighted sum of CCL-Bench metrics (lower is better when `minimize`).
Priority 1 — fix errors (crashes, timeouts). Priority 2 — improve the score.

## Design guidance

- Write general logic, not a lookup table; the policy must generalise to unseen workloads.
- Read `workload["config_space"]` to discover valid dimensions and their choices.
- Use `history` to learn from past failures (e.g. OOM with a certain tp value).
- You MUST call `submit_config` exactly once per iteration.
"""


# ── Workload card I/O ──────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_card(card: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(card, f, allow_unicode=True, sort_keys=False)


def _flatten_workload(card: dict, tuning: dict) -> dict:
    """Extract a flat workload dict from a CCL-Bench workload card + tuning config."""
    wl  = card.get("workload", {})
    mdl = wl.get("model", {})
    dat = wl.get("data", {})
    ex  = card.get("Model-executor", {})
    fw  = ex.get("framework", {})
    return {
        "name":          card.get("description", "run")[:60].strip(),
        "model_family":  mdl.get("model_family", ""),
        "phase":         mdl.get("phase", "training"),
        "moe":           mdl.get("moe", False),
        "precision":     mdl.get("precision", "fp32"),
        "num_layers":    mdl.get("num_layers"),
        "num_heads":     mdl.get("num_heads"),
        "num_kv_heads":  mdl.get("num_kv_heads"),
        "num_params":    mdl.get("num_params"),
        "batch_size":    dat.get("batch_size"),
        "seq_len":       dat.get("seq_len"),
        "run_script":    fw.get("run_script"),
        "trace_dir":     fw.get("trace_dir"),
        "config_space":  tuning.get("config_space", []),
    }


def _flatten_environment(card: dict) -> dict:
    """Extract a flat environment dict from a CCL-Bench workload card."""
    hw  = card.get("workload", {}).get("hardware", {})
    xpu = hw.get("xpu_spec", {})
    net = hw.get("network_topo", {})
    bws = net.get("bandwidth_gbps", [])
    ex  = card.get("Model-executor", {})
    fw  = ex.get("framework", {})
    return {
        "gpu_model":                 xpu.get("model", ""),
        "gpu_memory_gb":             xpu.get("memory_gb"),
        "total_gpus":                xpu.get("total_count", 1),
        "gpus_per_node":             xpu.get("count_per_node", 1),
        "intra_node_bandwidth_gbps": bws[0] if len(bws) > 0 else None,
        "inter_node_bandwidth_gbps": bws[1] if len(bws) > 1 else None,
        "network_topology":          net.get("topology", ""),
        "driver_version":            hw.get("driver_version", ""),
        "framework":                 fw.get("name", ""),
        "framework_version":         fw.get("version", ""),
    }


# ── generate_config program helpers ───────────────────────────────────────────

def load_gc(path: Path):
    spec = importlib.util.spec_from_file_location("gc_module", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_gc(code: str, version: int) -> Path:
    GC_DIR.mkdir(parents=True, exist_ok=True)
    path = GC_DIR / f"generate_config_v{version}.py"
    path.write_text(code)
    return path


# ── Execute ────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    workload: dict
    config: dict
    status: str            # "success" | "error" | "timeout"
    error_msg: str | None = None
    trace_dir: str | None = None


def execute(workload: dict, config: dict, timeout: int = 3600) -> RunResult:
    """Run workload's run_script with config choices injected as env vars."""
    run_script = workload.get("run_script")
    if not run_script:
        return RunResult(workload, config, "error",
                         error_msg="workload card missing 'run_script'")

    trace_dir = workload.get("trace_dir",
                             f"/tmp/ccl_bench_traces/{workload.get('name', 'run')}/")

    env = os.environ.copy()
    for k, v in config.items():
        env[k.upper()] = str(v)
    env["TRACE_DIR"]     = trace_dir
    env["WORKLOAD_NAME"] = workload.get("name", "")

    try:
        proc = subprocess.run(
            ["bash", run_script], env=env,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return RunResult(workload, config, "timeout",
                         error_msg=f"timed out after {timeout}s")
    except FileNotFoundError:
        return RunResult(workload, config, "error",
                         error_msg=f"run_script not found: {run_script}")

    if proc.returncode != 0:
        return RunResult(workload, config, "error",
                         error_msg=f"exit {proc.returncode}: {proc.stderr[-600:]}")

    return RunResult(workload, config, "success", trace_dir=trace_dir)


# ── Compute metric ─────────────────────────────────────────────────────────────

def compute_metric(run_result: RunResult, optimization_goal: dict) -> dict:
    """Apply CCL-Bench tool scripts to collected traces → objective score."""
    if run_result.status != "success":
        return {"status": run_result.status, "error_msg": run_result.error_msg,
                "metrics": {}, "score": float("inf")}

    direction   = optimization_goal.get("direction", "minimize")
    metric_cfgs = optimization_goal.get("metrics", [])
    values: dict[str, float] = {}

    for mc in metric_cfgs:
        name = mc["name"]
        try:
            proc = subprocess.run(
                [sys.executable, str(TOOLS_MAIN),
                 "--trace", run_result.trace_dir, "--metric", name],
                capture_output=True, text=True, timeout=120,
                cwd=str(TOOLS_MAIN.parent),
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr[-400:])
            values[name] = float(proc.stdout.strip())
        except Exception as e:
            return {"status": "metric_error",
                    "error_msg": f"metric '{name}' failed: {e}",
                    "metrics": values, "score": float("inf")}

    score = sum(mc.get("weight", 1.0) * values[mc["name"]] for mc in metric_cfgs)
    if direction == "maximize":
        score = -score

    return {"status": "success", "metrics": values, "score": score, "error_msg": None}


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _eval_score(history: list[dict]) -> tuple[float, float]:
    """(success_rate, mean_score). Higher success_rate is better; lower score is better."""
    ok = [r for r in history if r.get("status") == "success"]
    sr = len(ok) / len(history) if history else 0.0
    ms = sum(r["score"] for r in ok) / len(ok) if ok else float("inf")
    return (sr, ms)


def _is_better(new: tuple[float, float], old: tuple[float, float]) -> bool:
    sr_n, ms_n = new
    sr_o, ms_o = old
    if sr_n > sr_o:
        return True
    return sr_n == sr_o and ms_n < ms_o


# ── Single-run evaluation ──────────────────────────────────────────────────────

def run_once(gc_path: Path, workload: dict, environment: dict,
             card: dict, card_path: Path, tuning: dict,
             history: list[dict]) -> dict:
    """Execute one generate_config → execute → compute_metric cycle.

    Appends the run record to the workload card file and returns it.
    """
    mod = load_gc(gc_path)

    try:
        config = mod.generate_config(workload=workload, environment=environment,
                                     history=history)
    except Exception as e:
        record = {"config": {}, "metrics": {}, "score": float("inf"),
                  "status": "error", "error_msg": f"generate_config raised: {e}",
                  "trace_dir": None}
        _append_run(card, config, record)
        save_card(card, card_path)
        return record

    print("    config → " + "  ".join(f"{k}={v}" for k, v in sorted(config.items())))

    run_result    = execute(workload, config)
    metric_result = compute_metric(run_result, tuning["optimization_goal"])

    record = {
        "config":    config,
        "metrics":   metric_result.get("metrics", {}),
        "score":     metric_result.get("score"),
        "status":    metric_result.get("status"),
        "error_msg": metric_result.get("error_msg"),
        "trace_dir": run_result.trace_dir,
    }

    if metric_result["status"] == "success":
        m_str = "  ".join(f"{k}={v:.4g}" for k, v in metric_result["metrics"].items())
        print(f"    score={metric_result['score']:.4g}  ({m_str})")
    else:
        print(f"    {metric_result['status']}: {(metric_result.get('error_msg') or '')[:100]}")

    _append_run(card, config, record)
    save_card(card, card_path)
    return record


def _append_run(card: dict, config: dict, record: dict) -> None:
    if "runs" not in card or card["runs"] is None:
        card["runs"] = []
    card["runs"].append(record)


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


def _build_iteration_message(gc_code: str, history: list[dict],
                              iteration: int, max_iterations: int,
                              best_score: tuple[float, float],
                              workload: dict, environment: dict,
                              optimization_goal: dict) -> str:
    b_sr, b_ms = best_score
    env_str = "\n".join(f"  {k}: {v}" for k, v in environment.items() if v is not None)
    cs_str  = "\n".join(
        f"  {c['key']} ({c['type']}) choices={c.get('choices','?')} — {c.get('description','')}"
        for c in workload.get("config_space", [])
    )
    goal_str = (f"  direction: {optimization_goal.get('direction','minimize')}\n" +
                "\n".join(f"  - {mc['name']} (weight={mc.get('weight',1.0)})"
                          for mc in optimization_goal.get("metrics", [])))
    recent_json = json.dumps(history[-20:], indent=2, default=str)
    if len(recent_json) > 5000:
        recent_json = recent_json[:5000] + "\n... (truncated)"

    return (
        f"## Iteration {iteration}/{max_iterations}\n\n"
        f"## Workload\n"
        f"  {workload.get('model_family','?')} [{workload.get('phase','?')}]  "
        f"batch={workload.get('batch_size','?')}  seq={workload.get('seq_len','?')}  "
        f"precision={workload.get('precision','?')}\n\n"
        f"## Environment\n{env_str}\n\n"
        f"## Config Space\n{cs_str}\n\n"
        f"## Optimization Objective\n{goal_str}\n\n"
        f"## Current generate_config\n```python\n{gc_code}\n```\n\n"
        f"## Execution History\n"
        f"```\n{_format_history_table(history, optimization_goal)}\n```\n\n"
        f"```json\n{recent_json}\n```\n\n"
        f"Best so far:  success_rate={b_sr:.0%},  score={b_ms:.4g}\n\n"
        f"**Task:** fix failures first, then improve the score. "
        f"Submit via `submit_config`.\n"
    )


# ── CSV logging ────────────────────────────────────────────────────────────────

def _init_csv() -> None:
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "iteration", "version", "config", "metrics",
            "score", "status", "error_msg", "best_score",
        ])


def _log_csv(iteration: int, version: int, record: dict, best_score: float) -> None:
    with open(RESULTS_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            iteration, version,
            json.dumps(record.get("config", {})),
            json.dumps(record.get("metrics", {})),
            record.get("score", ""),
            record.get("status", ""),
            record.get("error_msg", ""),
            best_score,
        ])


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(card_path: Path, tuning_path: Path, seed_path: Path,
              max_iterations: int = 15, patience: int = 5) -> Path:
    """ADRS config tuning loop. Returns path of the best generate_config program."""
    api_key = API_KEY_FILE.read_text().strip()
    client  = anthropic.Anthropic(api_key=api_key)

    submit_tool = {
        "name": "submit_config",
        "description": (
            "Submit a revised `generate_config` function. It will be executed "
            "immediately; results appear next iteration. "
            "Must define `generate_config(workload, environment, history) -> dict`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python module defining `generate_config`.",
                },
                "rationale": {
                    "type": "string",
                    "description": "What changed and why.",
                },
            },
            "required": ["code"],
        },
    }

    card        = load_yaml(card_path)
    tuning      = load_yaml(tuning_path)
    workload    = _flatten_workload(card, tuning)
    environment = _flatten_environment(card)
    goal        = tuning["optimization_goal"]

    current_path = seed_path
    current_code = seed_path.read_text()
    history: list[dict] = []
    version = 0

    # ── Seed run ──────────────────────────────────────────────────────────
    print("\n[eval] Seed generate_config...")
    seed_record = run_once(current_path, workload, environment,
                           card, card_path, tuning, history)
    history.append(seed_record)

    best_score = _eval_score(history)
    best_path  = current_path
    no_improve = 0

    _init_csv()
    _log_csv(0, 0, seed_record, best_score[1])

    sr, ms = best_score
    print(f"\n[agent] Seed: status={seed_record['status']}, score={ms:.4g}")
    print(f"[agent] Loop: max={max_iterations}, patience={patience}\n")

    # ── Main loop ──────────────────────────────────────────────────────────
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*65}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"{'='*65}")

        msg = _build_iteration_message(
            current_code, history, iteration, max_iterations,
            best_score, workload, environment, goal,
        )
        messages = [{"role": "user", "content": msg}]
        submitted = False

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

            version += 1
            gc_path  = save_gc(code, version)
            print(f"\n  [v{version}] {gc_path.name}")

            try:
                mod = load_gc(gc_path)
                if not callable(getattr(mod, "generate_config", None)):
                    raise AttributeError("no callable 'generate_config'")
            except Exception as e:
                messages.append({"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": tool_use.id,
                    "content": json.dumps({"error": f"load failed: {e}"}),
                }]})
                continue

            record = run_once(gc_path, workload, environment,
                              card, card_path, tuning, history)
            history.append(record)

            cur_score = _eval_score(history)
            improved  = _is_better(cur_score, best_score)
            if improved:
                best_score = cur_score
                best_path  = gc_path
                no_improve = 0
            else:
                no_improve += 1

            _log_csv(iteration, version, record, best_score[1])
            current_code = code
            submitted    = True

            tag = "IMPROVED" if improved else "no improvement"
            b_sr, b_ms = best_score
            print(f"    {tag} — best score={b_ms:.4g}")

            messages.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": tool_use.id,
                "content": _format_history_table(history[-10:], goal) + f"\n\n{tag.upper()}",
            }]})
            break

        if not submitted:
            print(f"\n[agent] No program submitted at iteration {iteration}, stopping.")
            break
        if no_improve >= patience:
            print(f"\n[agent] Early stop: {patience} non-improving iterations.")
            break

    print(f"\n[agent] Results CSV → {RESULTS_CSV}")
    print(f"[agent] Workload card → {card_path}")
    return best_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CCL-Bench ADRS Config Tuning Agent")
    parser.add_argument("--card",           default=str(DEFAULT_CARD))
    parser.add_argument("--tuning",         default=str(DEFAULT_TUNING))
    parser.add_argument("--seed",           default=str(DEFAULT_SEED))
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--patience",       type=int, default=5)
    args = parser.parse_args()

    card_path   = Path(args.card)
    tuning_path = Path(args.tuning)
    card        = load_yaml(card_path)
    tuning      = load_yaml(tuning_path)
    wl          = _flatten_workload(card, tuning)
    env         = _flatten_environment(card)
    goal        = tuning["optimization_goal"]

    print("=" * 65)
    print("CCL-Bench ADRS Configuration Tuning Agent")
    print("=" * 65)
    print(f"Card:       {card_path.name}")
    print(f"Tuning:     {tuning_path.name}")
    print(f"Workload:   {wl.get('model_family','?')}  [{wl.get('phase','?')}]")
    print(f"Hardware:   {env.get('total_gpus','?')} × {env.get('gpu_model','?')}")
    print(f"Objective:  {goal.get('direction','minimize')} "
          + " + ".join(f"{mc['name']}×{mc.get('weight',1)}"
                       for mc in goal.get("metrics", [])))
    print(f"Seed:       {args.seed}")
    print(f"Max iters:  {args.max_iterations}  patience={args.patience}")
    print("=" * 65)

    best = run_agent(
        card_path=card_path,
        tuning_path=tuning_path,
        seed_path=Path(args.seed),
        max_iterations=args.max_iterations,
        patience=args.patience,
    )
    print(f"\nBest → {best}\n")
    print(best.read_text())


if __name__ == "__main__":
    main()
