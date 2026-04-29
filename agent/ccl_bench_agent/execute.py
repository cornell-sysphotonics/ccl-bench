"""
execute step — ADRS agent loop, step 2.

Runs the workload's run_script with config key-value pairs injected as
uppercase environment variables (e.g. tp=4 → TP=4).

Interface:
    result = execute(workload, config, timeout=3600)
    result.status  → "success" | "error" | "timeout"
    result.trace_dir → path where traces were written (on success)
"""

import os
import subprocess
from dataclasses import dataclass


@dataclass
class RunResult:
    workload: dict
    config: dict
    status: str            # "success" | "error" | "timeout"
    error_msg: str | None = None
    trace_dir: str | None = None


def execute(workload: dict, config: dict, timeout: int = 3600,
            dest_trace_dir: str | None = None) -> RunResult:
    """Run workload's run_script with config choices injected as env vars."""
    run_script = workload.get("run_script")
    if not run_script:
        return RunResult(workload, config, "error",
                         error_msg="workload card missing 'run_script'")

    trace_dir = (dest_trace_dir or
                 workload.get("trace_dir",
                              f"/tmp/ccl_bench_traces/{workload.get('name', 'run')}/"))

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
