import json
import os
from typing import List, Tuple


def metric_cal(directory: str) -> float:
    """
    Calculate communication-computation overlap percentage from Kineto traces.

    This metric measures how much communication overlaps with computation in time,
    indicating efficient scheduling.

    Args:
        directory (str): Path to the directory containing Kineto trace files.

    Returns:
        float: Overlap percentage (0-100).
    """

    csv_candidates = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.startswith("cuda_gpu_trace") and name.endswith(".csv")
    ]
    json_candidates = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.startswith("kineto_trace_") and name.endswith(".json")
    ]
    trace_files = sorted(csv_candidates or json_candidates)
    if not trace_files:
        raise FileNotFoundError(f"No cuda_gpu_trace*.csv or kineto_trace_*.json found under {directory}")

    overlaps: List[float] = []

    for trace_file in trace_files:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)

        # Categorize events into communication and computation
        comm_events = []
        comp_events = []

        comm_kernels = ["nccl", "allreduce", "allgather", "reducescatter", "alltoall"]
        comp_kernels = ["gemm", "conv", "matmul", "attention"]

        for event in trace_data.get("traceEvents", []):
            cat = str(event.get("cat", "")).lower()
            if cat != "kernel":
                continue

            name = event.get("name", "").lower()
            ts = event.get("ts", 0)  # timestamp in microseconds
            dur = event.get("dur", 0)  # duration in microseconds

            if dur == 0:
                continue

            event_info = (ts, ts + dur)

            # Check if communication kernel
            if any(kw in name for kw in comm_kernels):
                comm_events.append(event_info)
            # Check if computation kernel
            elif any(kw in name for kw in comp_kernels):
                comp_events.append(event_info)

        if not comm_events or not comp_events:
            raise ValueError(f"No communication or computation kernel events found in {trace_file}")

        # Calculate overlap
        total_comm_time = sum(end - start for start, end in comm_events)
        overlap_time = 0.0

        for comm_start, comm_end in comm_events:
            for comp_start, comp_end in comp_events:
                # Calculate intersection
                overlap_start = max(comm_start, comp_start)
                overlap_end = min(comm_end, comp_end)

                if overlap_start < overlap_end:
                    overlap_time += (overlap_end - overlap_start)

        # Calculate overlap percentage
        if total_comm_time > 0:
            overlap_pct = (overlap_time / total_comm_time) * 100
        else:
            overlap_pct = 0.0
        overlaps.append(overlap_pct)

    return sum(overlaps) / len(overlaps)
