"""Pipeline Bubble metric calculation.

For pipeline parallel training, measures the idle time (bubble) between
microbatches at each pipeline stage.

Bubble ratio = sum_idle_time / total_iteration_time

Lower bubble ratio indicates better pipeline efficiency.
"""

import json
from pathlib import Path


def metric_cal(directory: str) -> float:
    """Calculate pipeline bubble ratio from trace data.

    Args:
        directory (str): Path to the trace directory containing trace files.

    Returns:
        float: Bubble ratio between 0 and 1.
    """
    # Find Kineto trace files for each rank
    rank_traces = _find_rank_traces(directory)

    if not rank_traces:
        print("Warning: No trace files found for bubble analysis")
        return 0.0

    # For single-rank or non-pipeline traces, check for PP markers
    return _calculate_bubble_ratio(rank_traces)


def _find_rank_traces(directory: str) -> dict[int, str]:
    """Find trace files organized by rank."""
    rank_traces = {}

    for path in Path(directory).iterdir():
        if path.name.startswith("kineto_trace") and path.name.endswith(".json"):
            # Try to extract rank from filename (e.g., kineto_trace_0.json)
            parts = path.name.replace(".json", "").split("_")
            try:
                rank = int(parts[-1])
            except (ValueError, IndexError):
                rank = 0  # Default to rank 0

            rank_traces[rank] = str(path)

    return rank_traces


def _calculate_bubble_ratio(rank_traces: dict[int, str]) -> float:
    """Calculate bubble ratio from rank traces.

    For pipeline parallelism, bubbles occur when a pipeline stage
    is waiting for activations from the previous stage.
    """
    all_bubbles = []
    total_times = []

    for trace_path in rank_traces.values():
        bubble_time, total_time = _analyze_single_trace(trace_path)
        if total_time > 0:
            all_bubbles.append(bubble_time)
            total_times.append(total_time)

    if not total_times:
        return 0.0

    # Average bubble ratio across ranks
    total_bubble = sum(all_bubbles)
    total_duration = sum(total_times)

    if total_duration <= 0:
        return 0.0

    return total_bubble / total_duration


def _analyze_single_trace(trace_path: str) -> tuple[float, float]:
    """Analyze a single trace file for pipeline bubbles.

    Returns:
        Tuple[float, float]: (bubble_time, total_time) in microseconds
    """
    try:
        with Path(trace_path).open() as f:
            data = json.load(f)

        events = data.get("traceEvents", [])
        if not events:
            return 0.0, 0.0

        # Find GPU kernel events
        kernel_intervals = []
        pp_markers = []

        # Look for pipeline-related markers
        pp_patterns = ["pp_", "pipeline", "stage", "microbatch", "p2p", "send", "recv"]

        for event in events:
            cat = event.get("cat", "")
            name = event.get("name", "").lower()
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)

            if ts <= 0:
                continue

            # Track all kernel activity
            if cat == "kernel" and dur > 0:
                kernel_intervals.append((ts, ts + dur))

            # Track pipeline markers
            if any(p in name for p in pp_patterns):
                pp_markers.append((ts, dur, name))

        if not kernel_intervals:
            return 0.0, 0.0

        # Sort intervals by start time
        kernel_intervals.sort(key=lambda x: x[0])

        # Calculate idle time (bubbles) between kernels
        min_ts = kernel_intervals[0][0]
        max_ts = kernel_intervals[-1][1]

        # Merge overlapping intervals to find actual busy time
        merged = _merge_intervals(kernel_intervals)

        total_time = max_ts - min_ts

        # Only count significant bubbles (> 1ms gap)
        # This filters out normal scheduling jitter
        significant_bubble = 0.0
        for i in range(len(merged) - 1):
            gap = merged[i + 1][0] - merged[i][1]
            if gap > 1000:  # > 1ms
                significant_bubble += gap

    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Error reading {trace_path}: {e}")
        return 0.0, 0.0
    else:
        return significant_bubble, total_time


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # Overlapping or adjacent, merge
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged
