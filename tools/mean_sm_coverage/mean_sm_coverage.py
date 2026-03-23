"""
Metric: mean_sm_coverage
Description: Average Streaming Multiprocessor (SM) coverage across all kernels,
             weighted by kernel duration.
Unit: Percentage (%)
Returns: Float between 0-100, or -1 if data unavailable

Supported trace types (dispatched via workload YAML):
  nsys — reads NSYS SQLite file (uses actual hardware grid dimensions)
  json — reads PyTorch-profiler JSON files; uses "blocks per SM" launch
         parameter as a proxy for SM coverage:
         coverage = min(blocks_per_sm, 1.0) × 100 %
         (1.0 = all SMs have at least one block)
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
    from nsys_utils import find_sqlite_file

    sqlite_path = find_sqlite_file(directory)
    if sqlite_path is None:
        print(f"Error: No .sqlite file found in {directory}", file=sys.stderr)
        return -1
    try:
        conn = sqlite3.connect(sqlite_path)
        device_info = pd.read_sql_query(
            "SELECT gpuId, numMultiprocessors FROM TARGET_INFO_CUDA_DEVICE", conn)
        if len(device_info) == 0 or pd.isna(device_info['numMultiprocessors'].iloc[0]):
            conn.close()
            return -1
        num_sms = device_info['numMultiprocessors'].iloc[0]
        if not num_sms:
            conn.close()
            return -1
        kernels = pd.read_sql_query(
            "SELECT gridX, gridY, gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL", conn)
        conn.close()
        if len(kernels) == 0:
            return -1
        blocks = kernels['gridX'] * kernels['gridY'] * kernels['gridZ']
        coverage = (blocks / num_sms).clip(upper=1.0) * 100
        return float(coverage.mean())
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


def _calc_json(directory: str) -> float:
    """
    Mean SM coverage proxy from PyTorch-profiler JSON files.

    PyTorch profiler records 'blocks per SM' = total_grid_blocks / num_SMs for
    each kernel launch.  A value >= 1.0 means every SM has at least one block
    (100 % SM coverage); values < 1.0 indicate that only a fraction of SMs are
    active.  We compute the duration-weighted mean across all kernels in all
    rank files.
    """
    json_files = select_json_files(directory)
    if not json_files:
        print(f"Error: No JSON files found in {directory}", file=sys.stderr)
        return -1

    weighted_sum = 0.0
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
                args = e.get("args") or {}
                bsm = args.get("blocks per SM")
                if bsm is None:
                    continue
                # coverage ∈ [0, 1]: fraction of SMs that have at least one block
                coverage = min(float(bsm), 1.0)
                weighted_sum += coverage * dur
                total_dur += dur

    if total_dur == 0:
        print(f"Error: No kernel data with SM info found in {directory}", file=sys.stderr)
        return -1

    return float((weighted_sum / total_dur) * 100.0)


def metric_cal(directory: str) -> float:
    trace_types = _get_trace_types(directory)
    if "nsys" in trace_types:
        return _calc_nsys(directory)
    elif "json" in trace_types:
        return _calc_json(directory)
    else:
        print(
            f"Error: unsupported trace types {trace_types} for mean_sm_coverage",
            file=sys.stderr,
        )
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mean_sm_coverage.py <trace_directory>")
        sys.exit(1)
    print(metric_cal(sys.argv[1]))
