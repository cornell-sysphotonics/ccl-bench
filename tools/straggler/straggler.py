import glob
import json
import os
from typing import Dict, Tuple


def metric_cal(directory: str) -> float:
    """
    Calculate the relative lag of the slowest device or process in a communication group.
    
    Args:
        directory (str): Path to the directory containing PyTorch ET trace JSON files.

    Returns:
        float: The straggler metric value, normalized to [0, 1].
    """

    trace_files = sorted(glob.glob(os.path.join(directory, "kineto_trace_*.json")))
    if not trace_files:
        print(f"No kineto_trace_<rank>.json files found under: {directory}")
        return 0.0

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
            name = event.get("name", "")
            if not name:
                continue

            if event.get("cat", "").lower() != "kernel":
                continue

            COMM_KERNEL_PREFIXES = tuple(
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

            lowered_name = name.lower()
            if not any(lowered_name.startswith(prefix) for prefix in _COMM_KERNEL_PREFIXES):
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

    valid_windows = [window for window in device_windows.values() if window[0] < window[1]]
    if len(valid_windows) <= 1:
        return 0.0

    global_start = min(window[0] for window in valid_windows)
    slowest_end = max(window[1] for window in valid_windows)
    fastest_end = min(window[1] for window in valid_windows)
    total_window = slowest_end - global_start
    if total_window <= 0:
        return 0.0

    straggler_metric = (slowest_end - fastest_end) / total_window
    return straggler_metric
