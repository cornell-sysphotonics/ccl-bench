import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import pandas as pd


DTYPE_BYTES = {
    "Float": 4,
    "Float32": 4,
    "Double": 8,
    "Float64": 8,
    "Half": 2,
    "Float16": 2,
    "BFloat16": 2,
    "BF16": 2,
    "Int": 4,
    "Int32": 4,
    "Int64": 8,
    "Long": 8,
    "Int16": 2,
    "Short": 2,
    "Int8": 1,
    "Byte": 1,
}


def load_events(trace_path: str) -> list:
    with open(trace_path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]
    if isinstance(data, list):
        return data
    return []


def is_nccl_collective_kernel(
    event: dict,
    kernel_prefixes: tuple[str, ...],
    collective_names: tuple[str, ...],
) -> bool:
    if event.get("ph") != "X" or event.get("cat") != "kernel":
        return False
    name = event.get("name", "")
    if any(name.startswith(prefix) for prefix in kernel_prefixes):
        return True
    return event.get("args", {}).get("Collective name", "") in collective_names


def extract_nccl_collective_events(
    trace_path: str,
    is_collective_kernel: Callable[[dict], bool],
) -> pd.DataFrame:
    rows = []
    for event in load_events(trace_path):
        if not is_collective_kernel(event):
            continue

        args = event.get("args", {})
        in_nelems = args.get("In msg nelems")
        dtype = args.get("dtype")
        group_size = args.get("Group size")
        dur = event.get("dur")

        if in_nelems is None or dtype is None or group_size is None or dur is None:
            continue
        if not isinstance(in_nelems, (int, float)) or in_nelems <= 0 or dur <= 0:
            continue

        elem_bytes = DTYPE_BYTES.get(dtype)
        if elem_bytes is None:
            continue

        rows.append({
            "name": event.get("name", ""),
            "ts": event.get("ts", 0),
            "dur_us": dur,
            "in_nelems": int(in_nelems),
            "dtype": dtype,
            "elem_bytes": elem_bytes,
            "bytes": int(in_nelems) * elem_bytes,
            "group_size": group_size,
            "pg_desc": args.get("Process Group Description", ""),
        })

    return pd.DataFrame(rows)


def extract_tpu_collective_events(
    trace_path: str,
    hlo_category: str,
    group_key: Callable[[dict], str],
) -> pd.DataFrame:
    events = [
        event for event in load_events(trace_path)
        if event.get("ph") == "X"
        and event.get("args", {}).get("hlo_category") == hlo_category
    ]
    if not events:
        return pd.DataFrame()

    pids_by_group: dict[str, set] = defaultdict(set)
    for event in events:
        pids_by_group[group_key(event)].add(event.get("pid"))
    group_sizes = {key: len(pids) for key, pids in pids_by_group.items()}

    rows = []
    for event in events:
        args = event["args"]
        bytes_accessed = args.get("bytes_accessed")
        device_duration_ps = args.get("device_duration_ps")
        if bytes_accessed is None or device_duration_ps is None:
            continue

        nbytes = int(bytes_accessed)
        dur_us = float(device_duration_ps) / 1e6
        if nbytes <= 0 or dur_us <= 0:
            continue

        key = group_key(event)
        rows.append({
            "name": event.get("name", ""),
            "ts": event.get("ts", 0),
            "dur_us": dur_us,
            "bytes": nbytes,
            "group_size": group_sizes.get(key, 1),
        })

    return pd.DataFrame(rows)


def add_bandwidth_utilization(
    df: pd.DataFrame,
    algo_factor_multiplier: Callable[[float], float],
    bandwidth: float = 600.0,
) -> pd.DataFrame:
    df = df.copy()
    n = df["group_size"]
    df["duration_s"] = df["dur_us"] / 1e6
    df["algo_factor"] = algo_factor_multiplier(n)
    df["data_size_GB"] = df["bytes"] / (2**30)
    df["effective bandwidth(GB/s)"] = (
        df["data_size_GB"] * df["algo_factor"] / df["duration_s"]
    )
    df["bandwidth utilization"] = df["effective bandwidth(GB/s)"] / bandwidth
    return df


def get_bandwidth_utilization_from_trace(
    trace_path: str,
    extractor: Callable[[str], pd.DataFrame],
    algo_factor_multiplier: Callable[[float], float],
    bandwidth: float = 600.0,
) -> pd.DataFrame:
    df = extractor(trace_path)
    if df.empty:
        return df
    return add_bandwidth_utilization(
        df,
        algo_factor_multiplier=algo_factor_multiplier,
        bandwidth=bandwidth,
    )


def metric_cal_for_collective(
    directory: str,
    collective_name: str,
    gpu_trace_parser: Callable[[str], pd.DataFrame],
    tpu_trace_parser: Callable[[str], pd.DataFrame],
) -> float:
    d = Path(directory)
    is_tpu_trace_dir = "tpu" in d.name.lower()
    trace_files = (
        sorted(d.glob("*.json"))
        if is_tpu_trace_dir
        else sorted(d.glob("rank*_trace.json")) or sorted(d.glob("kineto_trace_*.json"))
    )

    for trace_path in trace_files:
        try:
            df = (
                tpu_trace_parser(str(trace_path))
                if is_tpu_trace_dir
                else gpu_trace_parser(str(trace_path))
            )
        except Exception as e:
            print(f"error parsing {trace_path}: {e}")
            continue

        if not df.empty:
            return float(df["effective bandwidth(GB/s)"].median())

    print(f"No {collective_name} events found in {directory}")
    return float("nan")
