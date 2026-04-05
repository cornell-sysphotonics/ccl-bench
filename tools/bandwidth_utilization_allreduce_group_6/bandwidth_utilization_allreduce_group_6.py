import json
from pathlib import Path
import pandas as pd


DTYPE_BYTES = {
    "Float": 4, "Float32": 4,
    "Double": 8, "Float64": 8,
    "Half": 2, "Float16": 2,
    "BFloat16": 2, "BF16": 2,
    "Int": 4, "Int32": 4,
    "Int64": 8, "Long": 8,
    "Int16": 2, "Short": 2,
    "Int8": 1, "Byte": 1,
}



def _load_events(trace_path: str) -> list:
    with open(trace_path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]
    if isinstance(data, list):
        return data
    return []


def _is_allreduce_kernel(event: dict) -> bool:
    """Check if an event is an nccl:all_reduce kernel event."""
    if event.get("ph") != "X" or event.get("cat") != "kernel":
        return False
    name = event.get("name", "")
    args = event.get("args", {})
    collective_name = args.get("Collective name", "")
    if name.startswith("ncclDevKernel_AllReduce") or name.startswith("ncclKernel_AllReduce"):
        return True
    if collective_name in ("all_reduce", "allreduce"):
        return True
    return False


def _extract_allreduce_events(trace_path: str) -> pd.DataFrame:
    """Extract all reduce kernel events from a trace JSON file."""
    events = _load_events(trace_path)
    rows = []
    for e in events:
        if not _is_allreduce_kernel(e):
            continue
        args = e.get("args", {})
        in_nelems = args.get("In msg nelems")
        dtype = args.get("dtype")
        group_size = args.get("Group size")
        dur = e.get("dur")  # microseconds

        if in_nelems is None or dtype is None or group_size is None or dur is None:
            continue
        if not isinstance(in_nelems, (int, float)) or in_nelems <= 0:
            continue
        if dur <= 0:
            continue

        elem_bytes = DTYPE_BYTES.get(dtype)
        if elem_bytes is None:
            continue

        rows.append({
            "name": e.get("name", ""),
            "ts": e.get("ts", 0),
            "dur_us": dur,
            "in_nelems": int(in_nelems),
            "dtype": dtype,
            "elem_bytes": elem_bytes,
            "bytes": int(in_nelems) * elem_bytes,
            "group_size": group_size,
            "pg_desc": args.get("Process Group Description", ""),
        })

    return pd.DataFrame(rows)


def _get_bandwidth_utilization(df: pd.DataFrame, bandwidth: float = 600.0) -> pd.DataFrame:
    """Calculate bandwidth utilization for all_reduce events.

    For all_reduce, the algorithm factor is 2*(N-1)/N where N is the group size.
    Effective bandwidth = data_size * algo_factor / duration.
    Utilization = effective_bandwidth / peak_bandwidth.
    """
    df = df.copy()
    n = df["group_size"]
    df["duration_s"] = df["dur_us"] / 1e6
    # all_reduce algo factor: 2*(N-1)/N
    df["algo_factor"] = 2 * (n - 1) / n
    df["data_size_GB"] = df["bytes"] / (2**30)
    df["effective bandwidth(GB/s)"] = df["data_size_GB"] * df["algo_factor"] / df["duration_s"]
    df["bandwidth utilization"] = df["effective bandwidth(GB/s)"] / bandwidth
    return df


def _get_bandwidth_utilization_from_trace(trace_path: str, bandwidth: float = 600.0) -> pd.DataFrame:
    df = _extract_allreduce_events(trace_path)
    if df.empty:
        return df
    return _get_bandwidth_utilization(df, bandwidth=bandwidth)


def metric_cal(directory: str) -> float:
    """
    Calculate the median allreduce bandwidth (GB/s) from *trace.json files.

    Finds all nccl:all_reduce kernel events across all rank trace files in the
    directory and returns the median effective bandwidth in GB/s.

    Args:
        directory (str): The directory path containing the trace JSON files.

    Returns:
        float: The median allreduce bandwidth in GB/s, or float("nan") if no
               allreduce events are found.
    """
    # Try rank trace files until one parses successfully
    d = Path(directory)
    trace_files = sorted(d.glob("rank*_trace.json")) or sorted(d.glob("kineto_trace_*.json"))
    if not trace_files:
        print(f"No trace JSON found in {directory}")
        return float("nan")

    for trace_path in trace_files:
        try:
            df = _get_bandwidth_utilization_from_trace(str(trace_path))
        except Exception as e:
            print(f"error parsing {trace_path}, trying next rank: {e}")
            continue

        if df.empty:
            continue

        return float(df["effective bandwidth(GB/s)"].median())

    print(f"No nccl:all_reduce kernel events found in {directory}")
    return float("nan")
