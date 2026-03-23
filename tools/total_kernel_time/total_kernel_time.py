"""
Metric: total_kernel_time
Description: Total GPU kernel execution time summed across all kernels (in ms).
Unit: ms
Returns: Float >= 0, or -1 if data unavailable

Supported trace types (dispatched via workload YAML):
  nsys — delegates to total_kernel_time_group_9 (nsys stats kernsum)
  json — reads PyTorch-profiler JSON files (rank0_trace.json, …)
"""

import json
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from json_sampling import select_json_files


def _load_yaml(directory: str) -> dict:
    for fn in os.listdir(directory):
        if fn.endswith(".yaml"):
            with open(os.path.join(directory, fn)) as f:
                return yaml.safe_load(f) or {}
    return {}


def _get_trace_types(directory: str) -> list:
    return _load_yaml(directory).get("metric_source", {}).get("traces", [])


def _calc_nsys(directory: str) -> float:
    from nsys_utils import collect_nsys_traces, run_nsys_kernsum_csv, extract_csv_block, parse_kernsum_csv

    traces = collect_nsys_traces(directory)
    total_ns = 0
    for pth in traces:
        out = run_nsys_kernsum_csv(pth)
        rows = parse_kernsum_csv(extract_csv_block(out))
        total_ns += sum(r["total_ns"] for r in rows)
    return total_ns / 1e9  # ns → s


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
    Total kernel time from PyTorch-profiler JSON files.
    Sums kernel durations across all rank files and converts µs → ms.
    """
    json_files = select_json_files(directory)
    if not json_files:
        print(f"Error: No JSON files found in {directory}", file=sys.stderr)
        return -1

    total_us = 0.0
    any_data = False

    for path in json_files:
        for e in _load_json_events(path):
            if (
                isinstance(e, dict)
                and e.get("ph") == "X"
                and e.get("cat") == "kernel"
            ):
                dur = e.get("dur")
                if dur is not None:
                    total_us += float(dur)
                    any_data = True

    if not any_data:
        print(f"Error: No kernel data found in {directory}", file=sys.stderr)
        return -1

    return float(total_us / 1e6)   # µs → s


def metric_cal(directory: str) -> float:
    trace_types = _get_trace_types(directory)
    if "nsys" in trace_types:
        return _calc_nsys(directory)
    elif "json" in trace_types:
        return _calc_json(directory)
    else:
        print(
            f"Error: unsupported trace types {trace_types} for total_kernel_time",
            file=sys.stderr,
        )
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python total_kernel_time.py <trace_directory>")
        sys.exit(1)
    print(metric_cal(sys.argv[1]))
