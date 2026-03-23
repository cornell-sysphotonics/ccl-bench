"""
Metric: total_trace_time
Description: Wall-clock duration of the profiled trace window (first to last
             recorded event), in seconds.
Unit: s
Returns: Float >= 0, or -1 if data unavailable

Supported trace types (dispatched via workload YAML):
  json        — reads PyTorch-profiler JSON files (rank0_trace.json, …)
  json_tpu — reads TPU profiler Chrome-trace JSON
  nsys        — reads NSYS SQLite file
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
    Total trace time from PyTorch-profiler JSON files (seconds).
    Uses span of all GPU kernel events across all rank files.
    """
    _all_json = [fn for fn in os.listdir(directory) if fn.endswith(".json")]
    _kineto = [fn for fn in _all_json if fn.startswith("kineto_trace_")]
    json_files = sorted(os.path.join(directory, fn) for fn in (_kineto or _all_json))
    if not json_files:
        print(f"Error: No JSON files found in {directory}", file=sys.stderr)
        return -1

    global_min = float("inf")
    global_max = float("-inf")

    for path in json_files:
        for e in _load_json_events(path):
            if (
                isinstance(e, dict)
                and e.get("ph") == "X"
                and e.get("cat") == "kernel"
            ):
                ts = e.get("ts")
                dur = e.get("dur")
                if ts is None or dur is None:
                    continue
                ts = float(ts)
                global_min = min(global_min, ts)
                global_max = max(global_max, ts + float(dur))

    if global_min == float("inf"):
        print(f"Error: No kernel events found in {directory}", file=sys.stderr)
        return -1

    return (global_max - global_min) / 1e6   # µs → s


def _calc_tpu(directory: str) -> float:
    """
    Total trace time from TPU profiler Chrome-trace JSON (seconds).
    Uses span of all TPU device events.
    """
    _all_json = [fn for fn in os.listdir(directory) if fn.endswith(".json")]
    _kineto = [fn for fn in _all_json if fn.startswith("kineto_trace_")]
    json_files = sorted(os.path.join(directory, fn) for fn in (_kineto or _all_json))
    if not json_files:
        print(f"Error: No JSON files found in {directory}", file=sys.stderr)
        return -1

    try:
        with open(json_files[0], encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_files[0]}: {e}", file=sys.stderr)
        return -1

    events = data.get("traceEvents", []) if isinstance(data, dict) else []

    tpu_pids: set = set()
    for e in events:
        if (
            isinstance(e, dict)
            and e.get("ph") == "M"
            and e.get("name") == "process_name"
            and "/device:TPU:" in e.get("args", {}).get("name", "")
        ):
            tpu_pids.add(e["pid"])

    global_min = float("inf")
    global_max = float("-inf")

    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        if tpu_pids and e.get("pid") not in tpu_pids:
            continue
        ts = e.get("ts")
        dur = e.get("dur")
        if ts is None or dur is None:
            continue
        ts = float(ts)
        global_min = min(global_min, ts)
        global_max = max(global_max, ts + float(dur))

    if global_min == float("inf"):
        print(f"Error: No TPU device events found in {directory}", file=sys.stderr)
        return -1

    return (global_max - global_min) / 1e9   # ns → s


def _calc_nsys(directory: str) -> float:
    """
    Total trace time from NSYS SQLite file (seconds).
    Uses span of all GPU kernel events.
    """
    import sqlite3

    sqlite_path = None
    for fn in os.listdir(directory):
        if fn.endswith(".sqlite"):
            sqlite_path = os.path.join(directory, fn)
            break
    if not sqlite_path:
        print(f"Error: No .sqlite file found in {directory}", file=sys.stderr)
        return -1

    try:
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        cur.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL")
        row = cur.fetchone()
        conn.close()
        if row is None or row[0] is None:
            return -1
        # NSYS timestamps are in nanoseconds
        return (row[1] - row[0]) / 1e9   # ns → s
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return -1


def metric_cal(directory: str) -> float:
    trace_types = _get_trace_types(directory)
    if "nsys" in trace_types:
        return _calc_nsys(directory)
    elif "json" in trace_types:
        return _calc_json(directory)
    elif "json_tpu" in trace_types:
        return _calc_tpu(directory)
    else:
        print(
            f"Error: unsupported trace types {trace_types} for total_trace_time",
            file=sys.stderr,
        )
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python total_trace_time.py <trace_directory>")
        sys.exit(1)
    print(metric_cal(sys.argv[1]))
