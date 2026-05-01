import json
from collections import defaultdict
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


def _is_reducescatter_kernel(event: dict) -> bool:
    if event.get("ph") != "X" or event.get("cat") != "kernel":
        return False
    name = event.get("name", "")
    collective_name = event.get("args", {}).get("Collective name", "")
    if name.startswith("ncclDevKernel_ReduceScatter") or name.startswith("ncclKernel_ReduceScatter"):
        return True
    if collective_name in ("reduce_scatter", "reducescatter"):
        return True
    return False


def _extract_reducescatter_events(trace_path: str) -> pd.DataFrame:
    events = _load_events(trace_path)
    rows = []
    for e in events:
        if not _is_reducescatter_kernel(e):
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
    df = df.copy()
    n = df["group_size"]
    df["duration_s"] = df["dur_us"] / 1e6
    # reduce_scatter ring algo factor: (N-1)/N
    df["algo_factor"] = (n - 1) / n
    df["data_size_GB"] = df["bytes"] / (2**30)
    df["effective bandwidth(GB/s)"] = df["data_size_GB"] * df["algo_factor"] / df["duration_s"]
    df["bandwidth utilization"] = df["effective bandwidth(GB/s)"] / bandwidth
    return df


def _get_bandwidth_utilization_from_trace(trace_path: str, bandwidth: float = 600.0) -> pd.DataFrame:
    df = _extract_reducescatter_events(trace_path)
    if df.empty:
        return df
    return _get_bandwidth_utilization(df, bandwidth=bandwidth)


def _extract_reducescatter_tpu_events(trace_path: str) -> pd.DataFrame:
    events = _load_events(trace_path)
    rs_events = [
        e for e in events
        if e.get("ph") == "X" and e.get("args", {}).get("hlo_category") == "reduce-scatter"
    ]
    if not rs_events:
        return pd.DataFrame()

    name_pids: dict = defaultdict(set)
    for e in rs_events:
        name_pids[e.get("name", "")].add(e["pid"])
    name_group_size = {name: len(pids) for name, pids in name_pids.items()}

    rows = []
    for e in rs_events:
        args = e["args"]
        bytes_str = args.get("bytes_accessed")
        dur_ps = args.get("device_duration_ps")
        if bytes_str is None or dur_ps is None:
            continue
        nbytes = int(bytes_str)
        dur_us = float(dur_ps) / 1e6
        if nbytes <= 0 or dur_us <= 0:
            continue
        rows.append({
            "name": e.get("name", ""),
            "ts": e.get("ts", 0),
            "dur_us": dur_us,
            "bytes": nbytes,
            "group_size": name_group_size.get(e.get("name", ""), 1),
        })

    return pd.DataFrame(rows)


def _get_bandwidth_utilization_from_trace_tpu(trace_path: str, bandwidth: float = 600.0) -> pd.DataFrame:
    df = _extract_reducescatter_tpu_events(trace_path)
    if df.empty:
        return df
    return _get_bandwidth_utilization(df, bandwidth=bandwidth)


def metric_cal(directory: str) -> float:
    """
    Calculate the median reduce_scatter bandwidth (GB/s) from trace JSON files.

    Finds all reduce_scatter kernel events across rank trace files in the
    directory and returns the median effective bandwidth in GB/s.

    Args:
        directory (str): The directory path containing the trace JSON files.

    Returns:
        float: The median reduce_scatter bandwidth in GB/s, or float("nan") if
               no reduce_scatter events are found.
    """
    d = Path(directory)

    if "tpu" in d.name.lower():
        tpu_files = sorted(d.glob("*.json"))
        for trace_path in tpu_files:
            try:
                df = _get_bandwidth_utilization_from_trace_tpu(str(trace_path))
            except Exception as e:
                print(f"error parsing {trace_path}: {e}")
                continue
            if df.empty:
                continue
            return float(df["effective bandwidth(GB/s)"].median())
    else:
        gpu_files = sorted(d.glob("rank*_trace.json")) or sorted(d.glob("kineto_trace_*.json"))
        for trace_path in gpu_files:
            try:
                df = _get_bandwidth_utilization_from_trace(str(trace_path))
            except Exception as e:
                print(f"error parsing {trace_path}, trying next rank: {e}")
                continue
            if df.empty:
                continue
            return float(df["effective bandwidth(GB/s)"].median())

    print(f"No reduce_scatter events found in {directory}")
    return float("nan")
