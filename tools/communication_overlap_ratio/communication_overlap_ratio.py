"""
Metric: communication_overlap_ratio
Description: Fraction of communication time overlapped (hidden) by concurrent
             computation. Higher values indicate better pipelining.
Unit: Ratio (0-1)
Returns: Float between 0-1, or -1 if data unavailable

Supported trace types (dispatched via workload YAML):
  nsys  — delegates to communication_overlap_ratio_group_9 (NSYS SQLite)
  json  — reads PyTorch-profiler JSON files (rank0_trace.json, …)
"""

import json
import os
import re
import sys
import yaml


# ── Communication kernel patterns ────────────────────────────────────────────

_COMM_RE = re.compile(
    r"^nccl|ncclDevKernel|ncclKernel|"
    r"allreduce|allgather|reducescatter|broadcast|"
    r"cross[_\-]?device|all_reduce|all_gather|reduce_scatter|"
    r"sendrecv",
    re.IGNORECASE,
)


def _is_comm(name: str) -> bool:
    return bool(_COMM_RE.search(name))


# ── YAML helper ──────────────────────────────────────────────────────────────

def _load_yaml(directory: str) -> dict:
    for fn in os.listdir(directory):
        if fn.endswith(".yaml"):
            with open(os.path.join(directory, fn)) as f:
                return yaml.safe_load(f) or {}
    return {}


def _get_trace_types(directory: str) -> list:
    return _load_yaml(directory).get("metric_source", {}).get("traces", [])


# ── NSYS backend ─────────────────────────────────────────────────────────────

def _calc_nsys(directory: str) -> float:
    from communication_overlap_ratio_group_9.communication_overlap_ratio_group_9 import calculate_metric
    return calculate_metric(directory)


# ── JSON backend ──────────────────────────────────────────────────────────────

def _load_json_events(path: str):
    """Load traceEvents from a PyTorch-profiler JSON file; returns list or []."""
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
            partial = content[bracket:]
            data = None
            for suffix in (']}', ']}}}'):
                try:
                    data = json.loads(partial + suffix)
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


def _merge_intervals(pairs):
    """Merge sorted (start, end) pairs and return merged list."""
    if not pairs:
        return []
    merged = [list(pairs[0])]
    for s, e in pairs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _intersection_time(a_merged, b_merged):
    """Total intersection length between two sorted merged interval lists."""
    total = 0
    i = j = 0
    while i < len(a_merged) and j < len(b_merged):
        start = max(a_merged[i][0], b_merged[j][0])
        end = min(a_merged[i][1], b_merged[j][1])
        if start < end:
            total += end - start
        if a_merged[i][1] < b_merged[j][1]:
            i += 1
        else:
            j += 1
    return total


def _calc_json(directory: str) -> float:
    """
    Communication overlap ratio from PyTorch-profiler JSON files.
    For each rank file: fraction of comm time that overlaps with compute kernels.
    Returns mean across all ranks that have both comm and compute.
    """
    _all_json = [fn for fn in os.listdir(directory) if fn.endswith(".json")]
    _kineto = [fn for fn in _all_json if fn.startswith("kineto_trace_")]
    json_files = sorted(os.path.join(directory, fn) for fn in (_kineto or _all_json))
    if not json_files:
        print(f"Error: No JSON files found in {directory}", file=sys.stderr)
        return -1

    per_rank = []
    for path in json_files:
        events = _load_json_events(path)
        comm_ivs = []
        compute_ivs = []
        for e in events:
            if (
                isinstance(e, dict)
                and e.get("ph") == "X"
                and e.get("cat") == "kernel"
            ):
                ts = e.get("ts")
                dur = e.get("dur")
                if ts is None or dur is None:
                    continue
                start = float(ts)
                end = start + float(dur)
                if _is_comm(e.get("name", "")):
                    comm_ivs.append((start, end))
                else:
                    compute_ivs.append((start, end))

        if not comm_ivs:
            continue

        comm_merged = _merge_intervals(sorted(comm_ivs))
        total_comm_time = sum(e - s for s, e in comm_merged)
        if total_comm_time == 0:
            continue

        if not compute_ivs:
            per_rank.append(0.0)
            continue

        compute_merged = _merge_intervals(sorted(compute_ivs))
        overlap = _intersection_time(comm_merged, compute_merged)
        per_rank.append(overlap / total_comm_time)

    if not per_rank:
        print(f"Error: No usable kernel data in {directory}", file=sys.stderr)
        return -1

    return float(sum(per_rank) / len(per_rank))


# ── Dispatcher ────────────────────────────────────────────────────────────────

def metric_cal(directory: str) -> float:
    """
    Calculate communication overlap ratio.

    Dispatches to the appropriate backend based on the workload YAML's
    metric_source.traces field.

    Args:
        directory: Path to the trace directory (must contain a workload YAML).

    Returns:
        float: Overlap ratio (0–1), or -1 if unavailable.
    """
    trace_types = _get_trace_types(directory)

    if "nsys" in trace_types:
        return _calc_nsys(directory)
    elif "json" in trace_types:
        return _calc_json(directory)
    else:
        print(
            f"Error: unsupported trace types {trace_types} for communication_overlap_ratio",
            file=sys.stderr,
        )
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python communication_overlap_ratio.py <trace_directory>")
        sys.exit(1)
    result = metric_cal(sys.argv[1])
    print(result)
