"""
Metric: moe_fraction
Description: Percentage of GPU time in Mixture of Experts kernels (expert compute,
             routing, gating, pplx dispatch/combine).
Unit: Percentage (%)
Returns: Float between 0-100, or -1 if data unavailable

Supported trace types (dispatched via workload YAML):
  nsys — reads NSYS SQLite file
  json — reads PyTorch-profiler JSON files (rank0_trace.json, …)
"""

import json
import os
import re
import sys
import yaml


# Kernel name patterns that identify MoE-related operations
_MOE_RE = re.compile(
    r"moe|expert|routing|gating|gate_|topk|fused_expert|"
    r"expert_kernel|dispatchkernel|combinekernel|"
    r"grouped_gemm|dispatch|combine",
    re.IGNORECASE,
)


def _load_yaml(directory: str) -> dict:
    for fn in os.listdir(directory):
        if fn.endswith(".yaml"):
            with open(os.path.join(directory, fn)) as f:
                return yaml.safe_load(f) or {}
    return {}


def _get_trace_types(directory: str) -> list:
    return _load_yaml(directory).get("metric_source", {}).get("traces", [])


def _calc_nsys(directory: str) -> float:
    from moe_fraction_group_9.moe_fraction_group_9 import calculate_metric
    return calculate_metric(directory)


def _load_json_events(path: str):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            idx = content.find('"traceEvents"')
            if idx == -1:
                return []
            bracket = content.find('[', idx)
            if bracket == -1:
                return []
            data = None
            for suffix in (']}', ']}}}'):
                try:
                    data = json.loads(content[bracket:] + suffix)
                    break
                except json.JSONDecodeError:
                    pass
            if data is None:
                return []
            if isinstance(data, list):
                return data
        except Exception:
            return []
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return []


def _calc_json(directory: str) -> float:
    """
    MoE fraction from PyTorch-profiler JSON files.
    Aggregates across all rank files: moe_kernel_time / total_kernel_time.
    """
    _all_json = [fn for fn in os.listdir(directory) if fn.endswith(".json")]
    _kineto = [fn for fn in _all_json if fn.startswith("kineto_trace_")]
    json_files = sorted(os.path.join(directory, fn) for fn in (_kineto or _all_json))
    if not json_files:
        print(f"Error: No JSON files found in {directory}", file=sys.stderr)
        return -1

    total_dur = 0.0
    moe_dur = 0.0
    any_data = False

    for path in json_files:
        for e in _load_json_events(path):
            if (
                isinstance(e, dict)
                and e.get("ph") == "X"
                and e.get("cat") == "kernel"
            ):
                dur = e.get("dur")
                if dur is None:
                    continue
                dur = float(dur)
                total_dur += dur
                any_data = True
                if _MOE_RE.search(e.get("name", "")):
                    moe_dur += dur

    if not any_data or total_dur == 0:
        print(f"Error: No kernel data found in {directory}", file=sys.stderr)
        return -1

    return float((moe_dur / total_dur) * 100.0)


def metric_cal(directory: str) -> float:
    trace_types = _get_trace_types(directory)
    if "nsys" in trace_types:
        return _calc_nsys(directory)
    elif "json" in trace_types:
        return _calc_json(directory)
    else:
        print(
            f"Error: unsupported trace types {trace_types} for moe_fraction",
            file=sys.stderr,
        )
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python moe_fraction.py <trace_directory>")
        sys.exit(1)
    print(metric_cal(sys.argv[1]))
