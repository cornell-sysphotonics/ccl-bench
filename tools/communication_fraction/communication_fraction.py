"""
Metric: communication_fraction
Description: Percentage of GPU/TPU time in communication kernels (NCCL, cross-device,
             P2P, XLA collectives). High values indicate communication bottlenecks.
Unit: Percentage (%)
Returns: Float between 0-100, or -1 if data unavailable

Supported trace types (dispatched via workload YAML):
  nsys        — reads the NSYS SQLite file produced by Nsight Systems
  json        — reads PyTorch-profiler JSON files (rank0_trace.json, …)
  tpu_profiler — reads TPU profiler Chrome-trace JSON (XLA collective ops)
"""

import json
import os
import re
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from json_sampling import select_json_files


# ── Communication kernel patterns ────────────────────────────────────────────

_COMM_RE = re.compile(
    r"nccl|allreduce|allgather|reducescatter|broadcast|"
    r"cross[_\-]?device|all_reduce|all_gather|reduce_scatter|"
    r"sendrecv|send|recv|p2p|communicate",
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
        comm_patterns = (
            'nccl|allreduce|allgather|reducescatter|broadcast|'
            'send|recv|p2p|cross_device|communicate|all_reduce|all_gather'
        )
        is_comm = kernels['kernel_name'].str.lower().str.contains(
            comm_patterns, na=False, regex=True)
        total_time = kernels['duration'].sum()
        comm_time = kernels[is_comm]['duration'].sum()
        if total_time == 0:
            return -1
        return float((comm_time / total_time) * 100)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return -1


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


_STEP_PATTERN = re.compile(r"ProfilerStep#(\d+)$")


def _calc_json(directory: str) -> float:
    """
    Communication fraction from PyTorch-profiler JSON files.
    For each rank file: computes per-step comm_kernel_time / total_kernel_time,
    trims first/last steps, then averages across inner steps and ranks.
    """
    json_files = select_json_files(directory)
    if not json_files:
        print(f"Error: No JSON files found in {directory}", file=sys.stderr)
        return -1

    per_rank = []
    for path in json_files:
        events = _load_json_events(path)

        # Collect step boundaries
        step_events = sorted(
            [e for e in events
             if isinstance(e, dict)
             and e.get("ph") == "X"
             and e.get("cat") == "user_annotation"
             and _STEP_PATTERN.match(e.get("name", ""))],
            key=lambda e: int(_STEP_PATTERN.match(e["name"]).group(1)),
        )

        # Collect kernel events
        kernels = []
        for e in events:
            if (
                isinstance(e, dict)
                and e.get("ph") == "X"
                and e.get("cat") == "kernel"
            ):
                dur = e.get("dur")
                ts = e.get("ts")
                if dur is None or ts is None:
                    continue
                kernels.append((float(ts), float(dur), _is_comm(e.get("name", ""))))

        if not kernels:
            continue

        if step_events:
            # Trim first and last steps
            inner = step_events[1:-1] if len(step_events) > 2 else step_events

            step_fractions = []
            for se in inner:
                s_start = float(se["ts"])
                s_end = s_start + float(se.get("dur", 0))
                total_dur = 0.0
                comm_dur = 0.0
                for ts, dur, is_c in kernels:
                    # Kernel overlaps with step window
                    if ts + dur > s_start and ts < s_end:
                        total_dur += dur
                        if is_c:
                            comm_dur += dur
                if total_dur > 0:
                    step_fractions.append((comm_dur / total_dur) * 100.0)

            if step_fractions:
                import statistics
                per_rank.append(statistics.mean(step_fractions))
        else:
            # No step events; compute over all kernels
            total_dur = sum(d for _, d, _ in kernels)
            comm_dur = sum(d for _, d, c in kernels if c)
            if total_dur > 0:
                per_rank.append((comm_dur / total_dur) * 100.0)

    if not per_rank:
        print(f"Error: No usable kernel data in {directory}", file=sys.stderr)
        return -1

    import statistics
    return float(statistics.mean(per_rank))


# ── TPU profiler backend ──────────────────────────────────────────────────────

# XLA collective operation category names used in TPU profiler traces
_TPU_COMM_OPS = {
    "all-reduce", "allreduce", "all_reduce",
    "all-gather", "allgather", "all_gather",
    "all-to-all", "alltoall",
    "reduce-scatter", "reducescatter", "reduce_scatter",
    "collective-permute", "collective-permute-start", "collective-permute-done",
    "send", "recv", "broadcast",
}


def _calc_tpu(directory: str) -> float:
    """
    Communication fraction from TPU profiler Chrome-trace JSON.
    Fraction of total TPU device time spent in XLA collective operations.
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

    # Identify TPU device PIDs from process_name metadata events
    tpu_pids: set = set()
    for e in events:
        if (
            isinstance(e, dict)
            and e.get("ph") == "M"
            and e.get("name") == "process_name"
            and "/device:TPU:" in e.get("args", {}).get("name", "")
        ):
            tpu_pids.add(e["pid"])

    # Per-device accumulation: comm time and total time
    device_total: dict = {}
    device_comm: dict = {}

    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        pid = e.get("pid")
        if tpu_pids and pid not in tpu_pids:
            continue

        # Prefer device_duration_ps (picoseconds → nanoseconds) for precision
        args = e.get("args", {}) or {}
        dev_ps = args.get("device_duration_ps")
        if dev_ps is not None:
            dur = float(dev_ps) / 1000.0   # ps → ns
        else:
            dur = float(e.get("dur") or 0)
        if dur <= 0:
            continue

        device_total[pid] = device_total.get(pid, 0.0) + dur

        name = (e.get("name") or "").lower()
        hlo_cat = (args.get("hlo_category") or "").lower()
        if name in _TPU_COMM_OPS or hlo_cat in _TPU_COMM_OPS:
            device_comm[pid] = device_comm.get(pid, 0.0) + dur

    if not device_total:
        print(f"Error: No TPU device events found in {directory}", file=sys.stderr)
        return -1

    # Average comm fraction across all TPU devices
    fractions = []
    for pid, total in device_total.items():
        if total > 0:
            comm = device_comm.get(pid, 0.0)
            fractions.append((comm / total) * 100.0)

    return float(sum(fractions) / len(fractions)) if fractions else -1


# ── Dispatcher ────────────────────────────────────────────────────────────────

def metric_cal(directory: str) -> float:
    """
    Calculate communication fraction.

    Dispatches to the appropriate backend based on the workload YAML's
    metric_source.traces field.

    Args:
        directory: Path to the trace directory (must contain a workload YAML).

    Returns:
        float: Communication fraction percentage (0–100), or -1 if unavailable.
    """
    trace_types = _get_trace_types(directory)

    if "nsys" in trace_types:
        return _calc_nsys(directory)
    elif "json" in trace_types:
        return _calc_json(directory)
    elif "tpu_profiler" in trace_types:
        return _calc_tpu(directory)
    else:
        print(
            f"Error: unsupported trace types {trace_types} for communication_fraction",
            file=sys.stderr,
        )
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python communication_fraction.py <trace_directory>")
        sys.exit(1)
    result = metric_cal(sys.argv[1])
    print(result)
