#!/usr/bin/env python3
"""
CCL-Bench ADRS Configuration Tuning Agent.

Iteratively synthesizes a Python program `generate_config` that maps workload
cards and environment descriptors to optimal configuration key-value pairs.

ADRS loop (each step is a standalone file):
  1. generate_config(workload, environment) → config dict   [generate_config.py]
  2. execute(workload, config)              → RunResult     [execute.py]
  3. compute_metric(run_result, goal)       → metric dict   [compute_metric.py]
  4. update_policy(gc_code, history, ...)   → new gc_code   [update_policy.py]
  5. update_history(history, record, ...)   → (in-place)    [update_history.py]

Run records are appended to the workload card after every execution so the
populated card (with traces) can be uploaded to the CCL-Bench platform.

Usage:
    python agent.py [--card PATH] [--tuning PATH] [--seed PATH]
                    [--max-iterations N] [--patience N]
"""

import argparse
import importlib.util
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic
import yaml

from execute import execute
from compute_metric import compute_metric
from update_policy import update_policy
from update_history import update_history
import run_cache


# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent
API_KEY_FILE = _HERE.parent / "API_KEY"
DEFAULT_CARD   = _HERE / "workload_card.yaml"
DEFAULT_TUNING = _HERE / "tuning_config.yaml"
DEFAULT_SEED   = _HERE / "generate_config.py"
DEFAULT_OUTPUT = _HERE / "runs"
_TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")

# These are set by _init_output_dir() once the output dir is known.
RUN_DIR     : Path
GC_DIR      : Path
RESULTS_CSV : Path
AGENT_LOG   : Path


def _init_output_dir(output_root: Path) -> None:
    """Create a timestamped run directory and set all output path globals."""
    global RUN_DIR, GC_DIR, RESULTS_CSV, AGENT_LOG
    RUN_DIR     = output_root / _TIMESTAMP
    GC_DIR      = RUN_DIR / "policies"
    RESULTS_CSV = RUN_DIR / "results.csv"
    AGENT_LOG   = RUN_DIR / "agent.log"
    RUN_DIR.mkdir(parents=True, exist_ok=True)


class _Tee:
    """Mirrors writes to both the original stream and a log file."""

    def __init__(self, stream, log_path: Path):
        self._stream = stream
        self._log = open(log_path, "w", buffering=1)  # line-buffered

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def close(self) -> None:
        self._log.close()

    # Proxy everything else (isatty, fileno, …) to the original stream.
    def __getattr__(self, name: str):
        return getattr(self._stream, name)


# ── Workload card I/O ──────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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


# ── generate_config loader ─────────────────────────────────────────────────────

def load_gc(path: Path):
    spec = importlib.util.spec_from_file_location("gc_module", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _sentinel(direction: str) -> float:
    return float("inf") if direction == "minimize" else float("-inf")


def _is_better(new: float, old: float, direction: str) -> bool:
    return new < old if direction == "minimize" else new > old


def _save_traces(record: dict, iteration: int) -> None:
    """Copy traces from the current run into RUN_DIR/traces/{timestamp}_iter{N}."""
    src = record.get("trace_dir")
    if not src:
        return
    src_path = Path(src)
    if not src_path.is_dir():
        return
    dest = RUN_DIR / "traces" / f"{_TIMESTAMP}_iter{iteration}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src_path, dest)
    print(f"    [traces] saved → {dest}")


# ── Single-iteration runner ────────────────────────────────────────────────────

def run_once(gc_path: Path, workload: dict, environment: dict,
             tuning: dict) -> dict:
    """Steps 1-3: generate_config → execute → compute_metric → record."""
    mod = load_gc(gc_path)
    policy_code = gc_path.read_text()

    try:
        config = mod.generate_config(workload=workload, environment=environment)
    except Exception as e:
        return {"config": {}, "metrics": {}, "score": float("inf"),
                "status": "error", "error_msg": f"generate_config raised: {e}",
                "trace_dir": None, "policy": policy_code}

    print("    config → " + "  ".join(f"{k}={v}" for k, v in sorted(config.items())))

    cached = run_cache.get(workload, config)
    if cached is not None:
        print("    [cache hit] reusing stored result")
        return {**cached, "policy": policy_code}

    run_result    = execute(workload, config)
    metric_result = compute_metric(run_result, tuning["optimization_goal"])

    record = {
        "config":    config,
        "metrics":   metric_result["metrics"],
        "score":     metric_result["score"],
        "status":    metric_result["status"],
        "error_msg": metric_result["error_msg"],
        "trace_dir": run_result.trace_dir,
        "policy":    policy_code,
    }

    run_cache.put(workload, config, record)

    if metric_result["status"] == "success":
        m_str = "  ".join(f"{k}={v:.4g}" for k, v in metric_result["metrics"].items())
        print(f"    score={metric_result['score']:.4g}  ({m_str})")
    else:
        print(f"    {metric_result['status']}: {(metric_result.get('error_msg') or '')[:100]}")

    return record


# ── CSV logging ────────────────────────────────────────────────────────────────

def _init_csv() -> None:
    with open(RESULTS_CSV, "w", newline="") as f:
        import csv as _csv
        _csv.writer(f).writerow([
            "iteration", "version", "config", "metrics",
            "score", "status", "error_msg", "best_score", "search_time_s",
        ])


def _log_csv(
    iteration: int, version: int, record: dict,
    best_score: float, search_time_s: float | None = None,
) -> None:
    with open(RESULTS_CSV, "a", newline="") as f:
        import csv as _csv
        _csv.writer(f).writerow([
            iteration, version,
            json.dumps(record.get("config", {})),
            json.dumps(record.get("metrics", {})),
            record.get("score", ""),
            record.get("status", ""),
            record.get("error_msg", ""),
            best_score,
            f"{search_time_s:.1f}" if search_time_s is not None else "",
        ])


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(card_path: Path, tuning_path: Path, seed_path: Path,
              max_iterations: int = 15, patience: int = 5,
              output_root: Path | None = None) -> Path:
    """ADRS config tuning loop. Returns path of the best generate_config program."""
    _init_output_dir(output_root or DEFAULT_OUTPUT)
    tee = _Tee(sys.stdout, AGENT_LOG)
    sys.stdout = tee
    print(f"[agent] Run dir → {RUN_DIR}")

    try:
        return _run_agent(card_path, tuning_path, seed_path, max_iterations, patience)
    finally:
        sys.stdout = tee._stream
        tee.close()


def _run_agent(card_path: Path, tuning_path: Path, seed_path: Path,
               max_iterations: int = 15, patience: int = 5) -> Path:
    run_cache.load()
    api_key = API_KEY_FILE.read_text().strip()
    client  = anthropic.Anthropic(api_key=api_key)

    card        = load_yaml(card_path)
    tuning      = load_yaml(tuning_path)
    workload    = _flatten_workload(card, tuning)
    environment = _flatten_environment(card)
    direction   = tuning["optimization_goal"].get("direction", "minimize")

    current_path = seed_path
    current_code = seed_path.read_text()
    history: list[dict] = []
    version = 0

    # ── Seed run (steps 1-3 + 5) ──────────────────────────────────────────
    print("\n[eval] Seed generate_config...")
    seed_record = run_once(current_path, workload, environment, tuning)
    update_history(history, seed_record, card, card_path)          # step 5
    _save_traces(seed_record, 0)

    best_score = (seed_record["score"]
                  if seed_record["status"] == "success"
                  else _sentinel(direction))
    best_path  = current_path
    no_improve = 0

    _init_csv()
    _log_csv(0, 0, seed_record, best_score)

    print(f"\n[agent] Seed: status={seed_record['status']}, score={best_score:.4g}")
    print(f"[agent] Loop: max={max_iterations}, patience={patience}\n")

    # ── Main loop ──────────────────────────────────────────────────────────
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*65}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"{'='*65}")

        # step 4 — update policy
        t0 = time.monotonic()
        gc_path, code = update_policy(
            gc_code=current_code,
            gc_dir=GC_DIR,
            version=version + 1,
            history=history,
            workload=workload,
            environment=environment,
            tuning=tuning,
            client=client,
            iteration=iteration,
            max_iterations=max_iterations,
            best_score=best_score,
        )
        search_time_s = time.monotonic() - t0

        if gc_path is None:
            print(f"\n[agent] No program submitted at iteration {iteration}, stopping.")
            break

        version += 1

        # steps 1-3 — run the new policy
        record = run_once(gc_path, workload, environment, tuning)

        # step 5 — update history
        update_history(history, record, card, card_path)
        _save_traces(record, iteration)

        score    = record.get("score", _sentinel(direction))
        improved = (record["status"] == "success"
                    and _is_better(score, best_score, direction))
        if improved:
            best_score = score
            best_path  = gc_path
            no_improve = 0
        else:
            no_improve += 1

        _log_csv(iteration, version, record, best_score, search_time_s)
        current_code = code

        tag = "IMPROVED" if improved else "no improvement"
        print(f"    {tag} — best score={best_score:.4g}")

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
    parser.add_argument("--output-dir",     default=str(DEFAULT_OUTPUT),
                        help="Root directory for run outputs (default: runs/)")
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
        output_root=Path(args.output_dir),
    )
    print(f"\nBest → {best}\n")
    print(best.read_text())


if __name__ == "__main__":
    main()
