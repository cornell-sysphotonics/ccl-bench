"""Communication/Computation Overlap metric calculation.

Measures the overlap ratio between communication (NCCL) and computation (CUDA kernels)
on the GPU timeline.

Overlap = time_comm_and_comp_overlap / (time_comm_only + time_comm_and_comp_overlap)

A higher overlap indicates better hiding of communication latency behind computation.
"""

import json
from pathlib import Path


def metric_cal(directory: str) -> float:
    """Calculate communication/computation overlap ratio from trace data.

    Args:
        directory (str): Path to the trace directory containing trace files.

    Returns:
        float: Overlap ratio between 0 and 1 (or percentage if > 1).
    """
    # Find Kineto trace files
    for path in Path(directory).iterdir():
        if path.name.startswith("kineto_trace") and path.name.endswith(".json"):
            overlap = _calculate_overlap_from_trace(path)
            if overlap >= 0:
                return overlap

    print("Warning: Could not calculate overlap, no suitable traces found")
    return 0.0


def _calculate_overlap_from_trace(trace_path: Path) -> float:
    """Calculate overlap from Kineto Chrome trace.

    Returns overlap ratio (0-1) or -1 if unable to calculate.
    """
    try:
        with trace_path.open() as f:
            data = json.load(f)

        events = data.get("traceEvents", [])
        if not events:
            return -1

        # Separate communication and computation kernel events
        comm_intervals = []
        comp_intervals = []

        # NCCL kernel names that indicate communication
        nccl_patterns = [
            "ncclDevKernel",
            "ncclKernel",
            "nccl",
            "AllReduce",
            "ReduceScatter",
            "AllGather",
            "Broadcast",
            "Reduce",
            "SendRecv",
            "AllToAll",
        ]

        for event in events:
            # Only look at GPU kernel events
            cat = event.get("cat", "")
            if cat != "kernel":
                continue

            name = event.get("name", "")
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)

            if ts <= 0 or dur <= 0:
                continue

            # Check if this is a communication kernel
            is_comm = any(pattern.lower() in name.lower() for pattern in nccl_patterns)

            if is_comm:
                comm_intervals.append((ts, ts + dur))
            else:
                comp_intervals.append((ts, ts + dur))

        if not comm_intervals:
            print("Warning: No communication kernels found")
            return 0.0

        if not comp_intervals:
            print("Warning: No computation kernels found")
            return 0.0

        # Calculate overlap
        overlap_time = _calculate_interval_overlap(comm_intervals, comp_intervals)
        total_comm_time = sum(end - start for start, end in comm_intervals)

        if total_comm_time <= 0:
            return 0.0

        # Overlap ratio: what fraction of comm time overlaps with computation
        overlap_ratio = overlap_time / total_comm_time
        return float(min(overlap_ratio, 1.0))  # Cap at 1.0

    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Error reading {trace_path}: {e}")
        return -1


def _calculate_interval_overlap(
    intervals_a: list[tuple[float, float]], intervals_b: list[tuple[float, float]]
) -> float:
    """Calculate total overlap time between two sets of intervals.

    Uses a sweep line algorithm for efficiency.
    """
    if not intervals_a or not intervals_b:
        return 0.0

    # Create events: (time, event_type, set)
    # event_type: 1 = start, -1 = end
    events = []

    for start, end in intervals_a:
        events.append((start, 1, "a"))
        events.append((end, -1, "a"))

    for start, end in intervals_b:
        events.append((start, 1, "b"))
        events.append((end, -1, "b"))

    # Sort by time, with starts before ends at same time
    events.sort(key=lambda x: (x[0], -x[1]))

    overlap_time = 0.0
    active_a = 0
    active_b = 0
    last_time = 0.0

    for time, event_type, set_id in events:
        # If both sets were active, add overlap time
        if active_a > 0 and active_b > 0:
            overlap_time += time - last_time

        # Update active counts
        if set_id == "a":
            active_a += event_type
        else:
            active_b += event_type

        last_time = time

    return overlap_time
