import json
import os
from typing import List, Dict, Any
import argparse


# NCCL kernel name prefixes (same style as your coll_call_num metric)
NCCL_KERNEL_PREFIXES = [
    "ncclDevKernel_AllReduce",
    "ncclDevKernel_ReduceScatter",
    "ncclDevKernel_AllGather",
    "ncclDevKernel_Broadcast",
    "ncclDevKernel_Reduce",
    "ncclDevKernel_SendRecv",
]


def _load_trace_events(trace_file: str) -> List[Dict[str, Any]]:
    """
    Load Chrome trace events from a Kineto trace file.

    We expect either:
      {
        "traceEvents": [ ... ]
      }
    or a top-level list of events.
    """
    with open(trace_file, "r") as f:
        trace_data = json.load(f)

    events = trace_data.get("traceEvents")
    if events is None:
        # Fallback: some tools emit a top-level list
        if isinstance(trace_data, list):
            events = trace_data
        else:
            raise ValueError(f"Unrecognized trace format in {trace_file}")

    if not isinstance(events, list):
        raise ValueError(f"traceEvents is not a list in {trace_file}")

    return events


def _is_nccl_kernel(event: Dict[str, Any]) -> bool:
    """
    Heuristic: identify NCCL communication kernels by name prefix.

    Assumes:
      - event["cat"] == "kernel" for GPU kernels.
      - event["name"] is something like "ncclDevKernel_AllReduceRingLLKernel..."
    """
    if event.get("cat") != "kernel":
        return False

    name = event.get("name", "")
    for prefix in NCCL_KERNEL_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


def metric_cal(directory: str) -> float:
    """
    Calculate the fraction of GPU kernel time spent in NCCL communication
    versus total GPU kernel time (comm + compute).

    CommFraction = T_comm / (T_comm + T_compute)

    where:
      - T_comm is the sum of durations of NCCL kernels.
      - T_compute is the sum of durations of all other GPU kernels.

    Args:
        directory (str): Path to the directory containing 'kineto_trace_0.json'.

    Returns:
        float: communication fraction in [0, 1]. If there are no GPU kernels
               or no total time, raises a ValueError.
    """
    trace_file = os.path.join(directory, "kineto_trace_0.json")

    if not os.path.exists(trace_file):
        raise FileNotFoundError(f"Kineto trace file not found: {trace_file}")

    events = _load_trace_events(trace_file)

    comm_time_us = 0.0
    compute_time_us = 0.0

    # Iterate over all GPU kernel events
    for e in events:
        if e.get("ph") != "X":
            continue
        if e.get("cat") != "kernel":
            continue

        dur_us = float(e.get("dur", 0.0))
        if dur_us <= 0.0:
            continue

        if _is_nccl_kernel(e):
            comm_time_us += dur_us
        else:
            compute_time_us += dur_us

    total_gpu_time_us = comm_time_us + compute_time_us

    if total_gpu_time_us <= 0.0:
        raise ValueError(
            f"No GPU kernel time found in trace (comm + compute = 0) for {trace_file}"
        )

    comm_fraction = comm_time_us / total_gpu_time_us
    return comm_fraction


if __name__ == "__main__":
    trace_directory = None
    metric_name = None
    metric_cal_func = None

    parser = argparse.ArgumentParser(description="Process trace directory and metric name.")
    parser.add_argument("--trace", type=str, required=True, help="Path to the trace directory")

    args = parser.parse_args()

    trace_directory = args.trace

    metric = metric_cal(trace_directory)
    print(metric)