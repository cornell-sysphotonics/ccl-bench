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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from json_sampling import select_json_files


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
    import sqlite3
    import pandas as pd
    from nsys_utils import find_sqlite_file

    sqlite_path = find_sqlite_file(directory)
    if sqlite_path is None:
        print(f"Error: No .sqlite file found in {directory}", file=sys.stderr)
        return -1
    try:
        conn = sqlite3.connect(sqlite_path)
        strings = pd.read_sql_query("SELECT id, value FROM StringIds", conn)
        string_map = dict(zip(strings['id'], strings['value']))
        kernels = pd.read_sql_query("""
            SELECT (end - start) as duration, demangledName, shortName
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        """, conn)
        conn.close()
        if len(kernels) == 0:
            return -1
        kernels['kernel_name'] = (
            kernels['shortName'].map(string_map)
            .fillna(kernels['demangledName'].map(string_map))
            .fillna('Unknown')
        )
        moe_patterns = (
            r'moe|expert|routing|gating|\bgate\b|\btopk\b|'
            r'fused_expert|expert_kernel|dispatchkernel|combinekernel'
        )
        is_moe = kernels['kernel_name'].str.lower().str.contains(
            moe_patterns, na=False, regex=True)
        total_time = kernels['duration'].sum()
        moe_time = kernels[is_moe]['duration'].sum()
        if total_time == 0:
            return -1
        return float((moe_time / total_time) * 100)
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
    MoE fraction from PyTorch-profiler JSON files.
    Aggregates across all rank files: moe_kernel_time / total_kernel_time.
    """
    json_files = select_json_files(directory)
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
