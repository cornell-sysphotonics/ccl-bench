"""
Metric: aggregate_gpu_utilization
Description: Overall GPU utilization — merged active kernel time as percentage of
             trace duration. Accounts for idle time and gaps between kernel launches.
Unit: Percentage (%)
Returns: Float between 0-100, or -1 if data unavailable

Supported trace types (dispatched via workload YAML):
  nsys  — reads the NSYS SQLite file produced by Nsight Systems
  json  — reads PyTorch-profiler JSON files (rank0_trace.json, …)
"""

import json
import os
import sys
import yaml


# ── YAML helper ──────────────────────────────────────────────────────────────

def _load_yaml(directory: str) -> dict:
    for fn in os.listdir(directory):
        if fn.endswith(".yaml"):
            with open(os.path.join(directory, fn)) as f:
                return yaml.safe_load(f) or {}
    return {}


def _get_trace_types(directory: str) -> list:
    data = _load_yaml(directory)
    return data.get("metric_source", {}).get("traces", [])


# ── NSYS backend ─────────────────────────────────────────────────────────────

def _calc_nsys(directory: str) -> float:
    from aggregate_gpu_utilization_group_9.aggregate_gpu_utilization_group_9 import calculate_metric
    return calculate_metric(directory)


# ── JSON backend ──────────────────────────────────────────────────────────────

def _parse_json_kernels(path: str):
    """
    Parse a PyTorch-profiler JSON trace file and yield (start_us, end_us) for
    every GPU kernel event (cat == 'kernel', ph == 'X').
    Skips the file gracefully if it cannot be parsed.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Fallback: try to extract events up to the first parse error
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            # Find traceEvents array start
            idx = content.find('"traceEvents"')
            if idx == -1:
                return
            bracket = content.find('[', idx)
            if bracket == -1:
                return
            # Try to recover by appending closing brackets
            partial = content[bracket:]
            # Add ]} to close the array and object
            for suffix in (']}', ']}}', ']}}}'):
                try:
                    data = json.loads(partial + suffix)
                    break
                except json.JSONDecodeError:
                    data = None
            if data is None:
                return
            if isinstance(data, list):
                data = {"traceEvents": data}
        except Exception:
            return

    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    for e in events:
        if (
            isinstance(e, dict)
            and e.get("ph") == "X"
            and e.get("cat") == "kernel"
        ):
            ts = e.get("ts")
            dur = e.get("dur")
            if ts is not None and dur is not None:
                yield float(ts), float(ts) + float(dur)


def _merge_intervals(intervals):
    """Merge a list of (start, end) tuples and return total merged duration."""
    if not intervals:
        return 0.0, 0.0, 0.0
    intervals = sorted(intervals)
    merged = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    merged_time = sum(e - s for s, e in merged)
    global_start = merged[0][0]
    global_end = merged[-1][1]
    return merged_time, global_start, global_end


def _calc_json(directory: str) -> float:
    """
    Aggregate GPU utilization from PyTorch-profiler JSON files.
    Processes all rank*_trace.json (or *.json) files, computes utilization per
    rank, and returns the mean across ranks.
    """
    _all_json = [fn for fn in os.listdir(directory) if fn.endswith(".json")]
    _kineto = [fn for fn in _all_json if fn.startswith("kineto_trace_")]
    json_files = sorted(os.path.join(directory, fn) for fn in (_kineto or _all_json))
    if not json_files:
        print(f"Error: No JSON files found in {directory}", file=sys.stderr)
        return -1

    per_rank = []
    for path in json_files:
        intervals = list(_parse_json_kernels(path))
        if not intervals:
            continue
        merged_time, global_start, global_end = _merge_intervals(intervals)
        trace_duration = global_end - global_start
        if trace_duration <= 0:
            continue
        util = (merged_time / trace_duration) * 100.0
        per_rank.append(util)

    if not per_rank:
        print(f"Error: No usable kernel data in {directory}", file=sys.stderr)
        return -1

    return float(sum(per_rank) / len(per_rank))


# ── Dispatcher ────────────────────────────────────────────────────────────────

def metric_cal(directory: str) -> float:
    """
    Calculate aggregate GPU utilization.

    Dispatches to the appropriate backend based on the workload YAML's
    metric_source.traces field.

    Args:
        directory: Path to the trace directory (must contain a workload YAML).

    Returns:
        float: GPU utilization percentage (0–100), or -1 if unavailable.
    """
    trace_types = _get_trace_types(directory)

    if "nsys" in trace_types:
        return _calc_nsys(directory)
    elif "json" in trace_types:
        return _calc_json(directory)
    else:
        print(
            f"Error: unsupported trace types {trace_types} for aggregate_gpu_utilization",
            file=sys.stderr,
        )
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aggregate_gpu_utilization.py <trace_directory>")
        sys.exit(1)
    result = metric_cal(sys.argv[1])
    print(result)
