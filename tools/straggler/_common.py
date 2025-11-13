import glob
import json
import os
from typing import Dict, List, Tuple

_COMM_KERNEL_PREFIXES = tuple(
    prefix.lower()
    for prefix in (
        "ncclDevKernel_AllReduce",
        "ncclDevKernel_ReduceScatter",
        "ncclDevKernel_AllGather",
        "ncclDevKernel_Broadcast",
        "ncclDevKernel_Reduce",
        "ncclDevKernel_SendRecv",
        "ccl",
        "hccl",
    )
)


def _iter_trace_files(directory: str) -> List[str]:
    return sorted(glob.glob(os.path.join(directory, "kineto_trace_*.json")))


def collect_comm_windows(directory: str) -> Tuple[List[str], Dict[str, Tuple[float, float]]]:
    trace_files = _iter_trace_files(directory)
    device_windows: Dict[str, Tuple[float, float]] = {}

    for trace_file in trace_files:
        try:
            with open(trace_file, "r") as f:
                trace_data = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {trace_file}")
            continue
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {trace_file}")
            continue

        for event in trace_data.get("traceEvents", []):
            if event.get("cat", "").lower() != "kernel":
                continue

            name = event.get("name", "")
            lowered_name = name.lower()
            if not lowered_name or not any(
                lowered_name.startswith(prefix) for prefix in _COMM_KERNEL_PREFIXES
            ):
                continue

            start = event.get("ts")
            duration = event.get("dur")
            pid = event.get("pid")
            if start is None or duration is None or pid is None:
                continue

            end_time = start + duration
            key = f"{trace_file}:{pid}"
            current_start, current_end = device_windows.get(key, (float("inf"), float("-inf")))
            if start < current_start:
                current_start = start
            if end_time > current_end:
                current_end = end_time
            device_windows[key] = (current_start, current_end)

    return trace_files, device_windows
