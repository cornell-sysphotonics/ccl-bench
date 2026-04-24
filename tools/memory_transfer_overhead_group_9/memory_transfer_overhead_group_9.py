"""
Metric: memory_transfer_overhead
Description: Percentage of trace time spent in memory copy operations (memcpy). High values
             indicate memory transfer bottlenecks, often from PCIe transfers or inefficient
             data movement.
Unit: Percentage (%)
Returns: Float between 0-100, or -1 if data unavailable
"""

import sqlite3
import sys
import os


def find_sqlite_file(path):
    """Find SQLite file in directory or return path if it's already a .sqlite file"""
    # Convert to absolute path to avoid any relative path issues
    path = os.path.abspath(path)
    
    if os.path.isfile(path) and path.endswith('.sqlite'):
        return path
    
    if os.path.isdir(path):
        sqlite_files = [f for f in os.listdir(path) if f.endswith('.sqlite')]
        if len(sqlite_files) == 0:
            return None
        # Prefer non-profiling files
        non_profiling = [f for f in sqlite_files if 'profiling' not in f.lower()]
        if non_profiling:
            return os.path.abspath(os.path.join(path, non_profiling[0]))
        return os.path.abspath(os.path.join(path, sqlite_files[0]))
    
    return None


def calculate_metric(path):
    """
    Calculate metric from SQLite trace file.
    
    Args:
        path: Either a directory containing .sqlite file or direct path to .sqlite file
    
    Returns:
        float: Metric value, or -1 if calculation fails
    """
    # Find the SQLite file
    sqlite_path = find_sqlite_file(path)
    if sqlite_path is None:
        print(f"Error: No .sqlite file found in {path}", file=sys.stderr)
        return -1
    
    try:
        with sqlite3.connect(sqlite_path) as conn:
            cur = conn.cursor()

            cur.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_MEMCPY'
            """)
            if cur.fetchone() is None:
                return -1

            cur.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_KERNEL'
            """)
            if cur.fetchone() is None:
                return -1

            cur.execute("""
                SELECT COUNT(*), SUM(end - start), MIN(start), MAX(end)
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
            """)
            memcpy_count, total_memcpy_time, memcpy_start, memcpy_end = cur.fetchone()

            cur.execute("""
                SELECT COUNT(*), MIN(start), MAX(end)
                FROM CUPTI_ACTIVITY_KIND_KERNEL
            """)
            kernel_count, kernel_start, kernel_end = cur.fetchone()

        if not memcpy_count or not kernel_count:
            return -1

        # Use the full activity span including both kernels and memcpy.
        trace_start = min(kernel_start, memcpy_start)
        trace_end = max(kernel_end, memcpy_end)
        trace_duration = trace_end - trace_start

        if trace_duration == 0:
            return -1
        
        return float((total_memcpy_time / trace_duration) * 100)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return -1


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trace_metric_utils import load_yaml, get_trace_types, summarize_kineto_kernel_breakdown
from json_sampling import select_json_files

# Shared JSON loader (handles truncated files)
def _load_json_events(path):
    import json
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
            bracket = content.find("[", idx)
            if bracket == -1:
                return []
            partial = content[bracket:]
            for suffix in ("]}",  "]}}}"):
                try:
                    data = json.loads(partial + suffix)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                return []
            if isinstance(data, list):
                return data
        except Exception:
            return []
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return []


def _calc_json(directory: str) -> float:
    summary = summarize_kineto_kernel_breakdown(directory)
    if summary is None:
        return -1.0
    total_kernel_dur, breakdown = summary
    mem_transfer_dur = breakdown.get("memory_transfer", 0.0)
    if total_kernel_dur <= 0:
        return -1.0
    return round((mem_transfer_dur / total_kernel_dur) * 100.0, 4)


def _calc_json_tpu(directory: str) -> float:
    """
    TPU memory-transfer overhead via XLA copy-start.N / copy-done.N event pairs.

    Numerator:   union of DMA transfer intervals from device_offset_ps (picoseconds).
                 Copies run in parallel, so overlapping intervals are merged before summing
                 to avoid double-counting concurrent transfers.
    Denominator: total step execution time from '$core.py:331 step' dur values
                 (stored in microseconds; converted to ps for the ratio).
    """
    import re
    COPY_START_RE = re.compile(r"copy-start\.([\d]+)")
    COPY_DONE_RE  = re.compile(r"copy-done\.([\d]+)")
    STEP_EVENT    = "$core.py:331 step"

    json_files = select_json_files(directory)
    if not json_files:
        return -1.0

    intervals: list = []   # (start_ps, end_ps) for each completed copy
    total_step_us = 0.0

    for path in json_files:
        events = _load_json_events(path)
        pending: dict = {}   # op_id → device_start_ps

        for e in events:
            if not isinstance(e, dict) or e.get("ph") != "X":
                continue
            name = str(e.get("name", ""))
            args  = e.get("args", {}) or {}

            # Step events — accumulate execution time (dur in µs)
            if name == STEP_EVENT:
                dur = e.get("dur")
                if dur is not None:
                    total_step_us += float(dur)
                continue

            # copy-start / copy-done pairs — collect DMA intervals (device ps)
            dev_ps = args.get("device_offset_ps")
            if dev_ps is None:
                continue
            dev_ps = float(dev_ps)

            m = COPY_START_RE.search(name)
            if m:
                pending[m.group(1)] = dev_ps
                continue

            m = COPY_DONE_RE.search(name)
            if m and m.group(1) in pending:
                t0 = pending.pop(m.group(1))
                if dev_ps > t0:
                    intervals.append((t0, dev_ps))

    if total_step_us <= 0 or not intervals:
        return -1.0

    # Merge overlapping intervals and sum their durations (union)
    intervals.sort()
    merged_ps = 0.0
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged_ps += cur_end - cur_start
            cur_start, cur_end = s, e
    merged_ps += cur_end - cur_start

    # Convert step time to ps and compute ratio
    total_step_ps = total_step_us * 1e6   # µs × 1e6 ps/µs
    return round((merged_ps / total_step_ps) * 100.0, 4)


def metric_cal(directory: str) -> float:
    trace_types = get_trace_types(load_yaml(directory))
    if "nsys" in trace_types:
        return calculate_metric(directory)
    if "json_tpu" in trace_types:
        return _calc_json_tpu(directory)
    if "json" in trace_types:
        return _calc_json(directory)
    print(f"[memory_transfer_overhead] Unsupported trace types {trace_types}", file=sys.stderr)
    return -1.0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python memory_transfer_overhead_group_9.py <trace_directory_or_sqlite_file>")
        sys.exit(1)
    print(metric_cal(sys.argv[1]))
