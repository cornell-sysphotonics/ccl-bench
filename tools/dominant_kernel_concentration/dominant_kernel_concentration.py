"""
Metric: dominant_kernel_concentration
Description: Percentage of total GPU time spent in the single most time-consuming
             kernel. High values (>70%) indicate a bottleneck.
Unit: Percentage (%)
Returns: Float between 0-100, or -1 if data unavailable

Supported trace types (dispatched via workload YAML):
  nsys — reads NSYS SQLite file
  json — reads PyTorch-profiler JSON files (rank0_trace.json, …)
"""

import json
import os
import sys
import yaml


def _load_yaml(directory: str) -> dict:
    for fn in os.listdir(directory):
        if fn.endswith(".yaml"):
            with open(os.path.join(directory, fn)) as f:
                return yaml.safe_load(f) or {}
    return {}


def _get_trace_types(directory: str) -> list:
    return _load_yaml(directory).get("metric_source", {}).get("traces", [])


def _calc_nsys(directory: str) -> float:
    from dominant_kernel_concentration_group_9.dominant_kernel_concentration_group_9 import calculate_metric
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
    Dominant kernel concentration from PyTorch-profiler JSON files.
    Aggregates kernel durations by name across all rank files, returns
    the top kernel's share of total kernel time.
    """
    json_files = sorted(
        os.path.join(directory, fn)
        for fn in os.listdir(directory)
        if fn.endswith(".json")
    )
    if not json_files:
        print(f"Error: No JSON files found in {directory}", file=sys.stderr)
        return -1

    kernel_totals: dict = {}
    total_dur = 0.0

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
                name = e.get("name", "unknown")
                kernel_totals[name] = kernel_totals.get(name, 0.0) + dur
                total_dur += dur

    if total_dur == 0 or not kernel_totals:
        print(f"Error: No usable kernel data in {directory}", file=sys.stderr)
        return -1

    top_dur = max(kernel_totals.values())
    return float((top_dur / total_dur) * 100.0)


def metric_cal(directory: str) -> float:
    trace_types = _get_trace_types(directory)
    if "nsys" in trace_types:
        return _calc_nsys(directory)
    elif "json" in trace_types:
        return _calc_json(directory)
    else:
        print(
            f"Error: unsupported trace types {trace_types} for dominant_kernel_concentration",
            file=sys.stderr,
        )
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dominant_kernel_concentration.py <trace_directory>")
        sys.exit(1)
    print(metric_cal(sys.argv[1]))
