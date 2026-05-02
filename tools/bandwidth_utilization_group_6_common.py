import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import pandas as pd

from trace_metric_utils import load_yaml


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


def load_network_config(directory: str):
    """Parse bandwidth config from workload YAML in directory.

    Returns (count_per_node, inter_bw_GBs, intra_bw_GBs) or None if unavailable.
    bandwidth_gbps[0] = scale-out (inter-node), bandwidth_gbps[1] = scale-up (intra-node).
    Values in YAML are Gbps; converted to GB/s by dividing by 8.
    """
    yaml_data = load_yaml(directory)
    try:
        hw = yaml_data["workload"]["hardware"]
        count_per_node = hw["xpu_spec"]["count_per_node"]
        bw_gbps = hw["network_topo"]["bandwidth_gbps"]
        inter_bw = bw_gbps[0] / 8.0
        intra_bw = bw_gbps[1] / 8.0
        return count_per_node, inter_bw, intra_bw
    except (KeyError, IndexError, TypeError):
        return None


def is_intra_node(pg_ranks: list, count_per_node: int) -> bool:
    """Return True if all ranks in pg_ranks reside on the same node."""
    if not pg_ranks or count_per_node <= 0:
        return False
    return len({r // count_per_node for r in pg_ranks}) == 1


def add_bandwidth_column(
    df: pd.DataFrame,
    count_per_node: int,
    inter_bw: float,
    intra_bw: float,
) -> pd.DataFrame:
    """Add a bandwidth_GBs column selected per-row based on process group node membership."""
    df = df.copy()
    df["bandwidth_GBs"] = df["pg_ranks"].apply(
        lambda ranks: intra_bw if is_intra_node(ranks, count_per_node) else inter_bw
    )
    return df


def _parse_pg_ranks(value) -> list:
    """Normalize Process Group Ranks to a list of ints (field may be a JSON string or list)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return []
    if isinstance(value, list):
        return [int(r) for r in value]
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
            "pg_ranks": _parse_pg_ranks(args.get("Process Group Ranks", [])),
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


def add_collective_bandwidth(
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
    bw = df["bandwidth_GBs"] if "bandwidth_GBs" in df.columns else bandwidth
    df["bandwidth utilization"] = df["effective bandwidth(GB/s)"] / bw
    return df


def metric_cal_for_collective(
    directory: str,
    collective_name: str,
    gpu_extractor: Callable[[str], pd.DataFrame],
    tpu_extractor: Callable[[str], pd.DataFrame],
    algo_factor_multiplier: Callable[[float], float],
    bandwidth: float = 600.0,
    tpu_bandwidth: float = 800.0,
) -> float:
    d = Path(directory)

    is_tpu_trace_dir = "tpu" in d.name.lower()

    trace_files = (
        sorted(d.glob("*.json"))
        if is_tpu_trace_dir
        else sorted(d.glob("rank*_trace.json")) or sorted(d.glob("kineto_trace_*.json"))
    )

    network_config = None if is_tpu_trace_dir else load_network_config(directory)

    for trace_path in trace_files:
        try:
            if is_tpu_trace_dir:
                df = tpu_extractor(str(trace_path))
                if df.empty:
                    continue
                df = add_collective_bandwidth(df, algo_factor_multiplier, tpu_bandwidth)
            else:
                df = gpu_extractor(str(trace_path))
                if df.empty:
                    continue
                if network_config is not None and "pg_ranks" in df.columns:
                    count_per_node, inter_bw, intra_bw = network_config
                    df = add_bandwidth_column(df, count_per_node, inter_bw, intra_bw)
                df = add_collective_bandwidth(df, algo_factor_multiplier, bandwidth)
        except Exception as e:
            print(f"error parsing {trace_path}: {e}")
            continue

        return float(df["effective bandwidth(GB/s)"].median())

    print(f"No {collective_name} events found in {directory}")
    return float("nan")
