import json
import os
from typing import List, Dict, Any
import argparse


# NCCL kernel name prefixes
NCCL_KERNEL_PREFIXES = [
    "ncclDevKernel_AllReduce",
    "ncclDevKernel_ReduceScatter",
    "ncclDevKernel_AllGather",
    "ncclDevKernel_Broadcast",
    "ncclDevKernel_Reduce",
    "ncclDevKernel_SendRecv",
]

TRACE_PATH = "kineto_trace_0.json"

# Perlmutter NVLink bandwidth:
# 4 × 25 GB/s = 100 GB/s per GPU–GPU pair
LINK_BW_GBPS = 100.0
LINK_BW_BPS = LINK_BW_GBPS * 1e9

ALPHA = 0.0  # ring latency term (seconds)
MIN_BYTES = 0

DTYPE_BYTES = {
    "Float": 4, "Float32": 4,
    "Double": 8, "Float64": 8,
    "Half": 2, "Float16": 2,
    "BFloat16": 2, "BF16": 2,
}


def _load_events(trace_file: str) -> List[Dict[str, Any]]:
    with open(trace_file, "r") as f:
        trace = json.load(f)

    if isinstance(trace, dict) and "traceEvents" in trace:
        return trace["traceEvents"]
    elif isinstance(trace, list):
        return trace
    else:
        raise ValueError(f"Unrecognized trace format in {trace_file}")


def _expected_time(S_bytes: int, n: int) -> float:
    """
    T_exp = α(n-1) + (S/n)*(n-1)*(1/B)
    """
    if n <= 1:
        return 0.0
    beta = 1.0 / LINK_BW_BPS
    return ALPHA * (n - 1) + (S_bytes / n) * (n - 1) * beta


def metric_cal(directory: str) -> float:
    """
    Average bandwidth utilization for NCCL AllReduce kernels.

    Utilization = B_obs / B_exp
                = (S / T_obs) / (S / T_exp)
                = T_exp / T_obs

    Args:
        directory (str): Path to trace directory containing kineto_trace_0.json

    Returns:
        float: average bandwidth utilization (0–1+)

    Raises:
        FileNotFoundError: if trace file does not exist
        ValueError: if no valid kernels are found
    """
    trace_file = os.path.join(directory, TRACE_PATH)

    if not os.path.exists(trace_file):
        raise FileNotFoundError(f"Kineto trace file not found: {trace_file}")

    events = _load_events(trace_file)

    utils = []

    for e in events:
        if e.get("ph") != "X":
            continue
        if e.get("cat") != "kernel":
            continue

        name = e.get("name", "")
        if not name.startswith("ncclDevKernel_AllReduce"):
            continue

        args = e.get("args", {})

        if (
            "In msg nelems" not in args
            or "dtype" not in args
            or "Group size" not in args
        ):
            continue

        dtype = args["dtype"]
        if dtype not in DTYPE_BYTES:
            continue

        nelems = args["In msg nelems"]
        if not isinstance(nelems, (int, float)) or nelems <= 0:
            continue

        S_bytes = int(nelems) * DTYPE_BYTES[dtype]
        if S_bytes < MIN_BYTES:
            continue

        n = args["Group size"]

        T_obs = e.get("dur", 0.0) / 1e6  # us → s
        if T_obs <= 0:
            continue

        T_exp = _expected_time(S_bytes, n)
        if T_exp <= 0:
            continue

        util = T_exp / T_obs
        utils.append(util)

    if not utils:
        raise ValueError(
            f"No valid NCCL AllReduce kernels with size metadata found in {trace_file}"
        )

    return sum(utils) / len(utils)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bandwidth utilization metric")
    parser.add_argument("--trace", type=str, required=True, help="Path to trace directory")
    args = parser.parse_args()

    metric = metric_cal(args.trace)
    print(metric)