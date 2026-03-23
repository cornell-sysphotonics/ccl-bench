"""
Metric: load_imbalance_ratio
Description: Ratio of max to min GPU active time across GPUs (1.0 = perfectly balanced).
             High values indicate stragglers or uneven work distribution.
Unit: ratio
Returns: Float >= 1.0, or -1 if data unavailable (single GPU, or no data)

Supported trace types (dispatched via workload YAML):
  nsys — reads NSYS SQLite file (per-device kernel timings)
  json — reads PyTorch-profiler JSON files; one file per rank/GPU
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
    import sqlite3
    import pandas as pd
    import numpy as np
    from nsys_utils import find_sqlite_file

    sqlite_path = find_sqlite_file(directory)
    if sqlite_path is None:
        print(f"Error: No .sqlite file found in {directory}", file=sys.stderr)
        return -1
    try:
        conn = sqlite3.connect(sqlite_path)
        kernels = pd.read_sql_query(
            "SELECT deviceId, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL", conn)
        conn.close()
        if len(kernels) == 0:
            return -1
        unique_gpus = kernels['deviceId'].unique()
        if len(unique_gpus) <= 1:
            return -1
        gpu_times = []
        for gpu_id in unique_gpus:
            gk = kernels[kernels['deviceId'] == gpu_id]
            intervals = sorted(zip(gk['start'].values, gk['end'].values))
            merged_time = 0
            cur_start, cur_end = intervals[0]
            for s, e in intervals[1:]:
                if s <= cur_end:
                    cur_end = max(cur_end, e)
                else:
                    merged_time += cur_end - cur_start
                    cur_start, cur_end = s, e
            merged_time += cur_end - cur_start
            gpu_times.append(merged_time)
        gpu_times = np.array(gpu_times)
        if gpu_times.min() == 0:
            return -1
        return float(gpu_times.max() / gpu_times.min())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return -1


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


def _merge_total(intervals):
    """Return total merged duration from a list of (start, end) tuples."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    cs, ce = intervals[0]
    merged = 0.0
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged += ce - cs
            cs, ce = s, e
    merged += ce - cs
    return merged


def _calc_json(directory: str) -> float:
    """
    Load imbalance ratio from PyTorch-profiler JSON files.
    Each rank file represents one GPU. Computes total merged active kernel time
    per rank, then returns max_time / min_time.
    """
    json_files = select_json_files(directory)
    if not json_files:
        print(f"Error: No JSON files found in {directory}", file=sys.stderr)
        return -1

    per_rank_time = []

    for path in json_files:
        intervals = []
        for e in _load_json_events(path):
            if (
                isinstance(e, dict)
                and e.get("ph") == "X"
                and e.get("cat") == "kernel"
            ):
                ts = e.get("ts")
                dur = e.get("dur")
                if ts is not None and dur is not None:
                    intervals.append((float(ts), float(ts) + float(dur)))
        if intervals:
            per_rank_time.append(_merge_total(intervals))

    if len(per_rank_time) < 2:
        # Single GPU or insufficient data — no imbalance to measure
        return -1

    max_t = max(per_rank_time)
    min_t = min(per_rank_time)
    if min_t <= 0:
        return -1

    return float(max_t / min_t)


def metric_cal(directory: str) -> float:
    trace_types = _get_trace_types(directory)
    if "nsys" in trace_types:
        return _calc_nsys(directory)
    elif "json" in trace_types:
        return _calc_json(directory)
    else:
        print(
            f"Error: unsupported trace types {trace_types} for load_imbalance_ratio",
            file=sys.stderr,
        )
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_imbalance_ratio.py <trace_directory>")
        sys.exit(1)
    print(metric_cal(sys.argv[1]))
