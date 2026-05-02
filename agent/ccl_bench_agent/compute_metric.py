"""
compute_metric step — ADRS agent loop, step 3.

Invokes CCL-Bench tool scripts against the collected traces to produce a
scalar objective score.

Interface:
    result = compute_metric(run_result, optimization_goal)
    result["status"]   → "success" | "error" | "timeout" | "metric_error"
    result["metrics"]  → {metric_name: float, ...}
    result["score"]    → float
    result["error_msg"]→ str | None
"""

import sys
import subprocess
from pathlib import Path

from execute import RunResult


_HERE      = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
TOOLS_MAIN = _REPO_ROOT / "tools" / "main.py"


def compute_metric(run_result: RunResult, optimization_goal: dict) -> dict:
    """Apply CCL-Bench tool scripts to collected traces → objective score."""
    direction   = optimization_goal.get("direction", "minimize")
    bad_score   = float("inf") if direction == "minimize" else float("-inf")

    if run_result.status != "success":
        return {"status": run_result.status, "error_msg": run_result.error_msg,
                "metrics": {}, "score": bad_score}

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
                    "metrics": values, "score": bad_score}

    score = sum(mc.get("weight", 1.0) * values[mc["name"]] for mc in metric_cfgs)

    return {"status": "success", "metrics": values, "score": score, "error_msg": None}
