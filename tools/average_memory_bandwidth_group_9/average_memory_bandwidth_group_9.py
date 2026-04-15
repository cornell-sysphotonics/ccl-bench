"""
Metric: average_memory_bandwidth
Description: Average memory bandwidth achieved during memory copy operations. Lower values
             compared to hardware peak indicate inefficient memory access patterns or small
             transfer sizes.
Unit: GB/s (Gigabytes per second)
Returns: Float (GB/s), or -1 if data unavailable
"""

import sqlite3
import pandas as pd
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
        conn = sqlite3.connect(sqlite_path)
        
        # Check if memcpy table exists
        tables = pd.read_sql_query("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_MEMCPY'
        """, conn)
        
        if len(tables) == 0:
            conn.close()
            return -1
        
        # Load memory copy data
        memcpy = pd.read_sql_query("""
            SELECT 
                (end - start) as duration,
                bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
        """, conn)
        
        conn.close()
        
        if len(memcpy) == 0:
            return -1
        
        # Filter out invalid values (global avg: total bytes / total time)
        memcpy = memcpy[(memcpy['bytes'] > 0) & (memcpy['duration'] > 0)]
        
        if len(memcpy) == 0:
            return -1
        
        total_bytes = float(memcpy['bytes'].sum())
        total_duration_ns = float(memcpy['duration'].sum())
        
        if total_duration_ns <= 0:
            return -1

        # total_bytes / total_duration_ns → GB/s (ns cancels with bytes→GB factor)
        return float(total_bytes / total_duration_ns)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return -1


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trace_metric_utils import load_yaml, get_trace_types, _load_json_events, _MEMCPY_RE
from json_sampling import select_json_files

def _calc_json(directory: str) -> float:
    json_files = select_json_files(directory)
    if not json_files:
        return -1.0

    total_bytes = 0.0
    total_dur_us = 0.0

    for path in json_files:
        events = _load_json_events(path)
        for e in events:
            if not isinstance(e, dict) or e.get("ph") != "X":
                continue
            name = str(e.get("name", ""))
            if not _MEMCPY_RE.search(name):
                continue
            dur = e.get("dur")
            if not dur or float(dur) <= 0:
                continue
            args = e.get("args", {}) or {}
            # GPU kineto traces use "bytes"; TPU XLA traces use "bytes_accessed"
            raw = args.get("bytes") or args.get("bytes_accessed") or args.get("raw_bytes_accessed")
            if raw is not None:
                b = float(raw)
                if b > 0:
                    total_bytes += b
                    total_dur_us += float(dur)

    if total_dur_us <= 0:
        return -1.0

    # bytes / µs = MB/s  →  ÷ 1000 = GB/s
    return float((total_bytes / total_dur_us) / 1000.0)

def _calc_json_tpu(directory: str) -> float:
    """
    Compute TPU memory bandwidth by pairing copy-start.N / copy-done.N events.

    XLA emits copy-start (DMA kickoff, ~8 ns dur) and copy-done (barrier) as
    separate X-phase events.  The wall-clock 'dur' of copy-start is too small
    to be meaningful; the real transfer time is:
        copy-done.device_offset_ps – copy-start.device_offset_ps
    Dividing bytes_accessed by that gives a physically sensible bandwidth.
    """
    import re
    COPY_START_RE = re.compile(r"copy-start\.([\d]+)")
    COPY_DONE_RE  = re.compile(r"copy-done\.([\d]+)")

    json_files = select_json_files(directory)
    if not json_files:
        return -1.0

    total_bytes = 0.0
    total_dur_ps = 0.0

    for path in json_files:
        events = _load_json_events(path)
        pending: dict = {}          # op_id → (bytes, device_start_ps)

        for e in events:
            if not isinstance(e, dict) or e.get("ph") != "X":
                continue
            name = str(e.get("name", ""))
            args = e.get("args", {}) or {}
            dev_ps = args.get("device_offset_ps")
            if dev_ps is None:
                continue
            dev_ps = float(dev_ps)

            m = COPY_START_RE.search(name)
            if m:
                raw = args.get("bytes_accessed") or args.get("raw_bytes_accessed")
                if raw:
                    pending[m.group(1)] = (float(raw), dev_ps)
                continue

            m = COPY_DONE_RE.search(name)
            if m and m.group(1) in pending:
                b, t0 = pending.pop(m.group(1))
                dur_ps = dev_ps - t0
                if dur_ps > 0 and b > 0:
                    total_bytes += b
                    total_dur_ps += dur_ps

    if total_dur_ps <= 0:
        return -1.0

    # bytes / ps × 1000 = GB/s  (1 byte/ps = 1 TB/s, ÷1000 = GB/s)
    return float((total_bytes / total_dur_ps) * 1000.0)


def metric_cal(directory: str) -> float:
    trace_types = get_trace_types(load_yaml(directory))
    if "nsys" in trace_types:
        return calculate_metric(directory)
    if "json" in trace_types:
        return _calc_json(directory)
    if "json_tpu" in trace_types:
        return _calc_json_tpu(directory)
    print(f"[average_memory_bandwidth] Unsupported trace types {trace_types}", file=sys.stderr)
    return -1.0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python average_memory_bandwidth_group_9.py <trace_directory_or_sqlite_file>")
        sys.exit(1)
    print(metric_cal(sys.argv[1]))
