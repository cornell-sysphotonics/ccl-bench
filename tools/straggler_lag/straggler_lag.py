"""Straggler Lag metric calculation.

Measures the relative lag of the slowest device/process in a communication group.
This indicates load imbalance and synchronization overhead.

Straggler lag = (max_end_time - min_end_time) / avg_iteration_time

Higher values indicate more imbalance.
"""

import json
from pathlib import Path


def metric_cal(directory: str) -> float:
    """Calculate straggler lag from multi-rank trace data.

    Args:
        directory (str): Path to the trace directory containing trace files.

    Returns:
        float: Normalized straggler lag (0 = perfect balance).
    """
    # Find all rank traces
    rank_traces = _find_rank_traces(directory)

    if len(rank_traces) <= 1:
        print("Warning: Single rank or no traces found, straggler lag = 0")
        return 0.0

    # Analyze iteration end times per rank
    rank_end_times = {}
    iteration_times = []

    for rank, trace_path in rank_traces.items():
        end_time, iter_time = _analyze_rank_trace(trace_path)
        if end_time > 0:
            rank_end_times[rank] = end_time
            if iter_time > 0:
                iteration_times.append(iter_time)

    if len(rank_end_times) <= 1:
        return 0.0

    # Calculate straggler lag
    end_times = list(rank_end_times.values())
    max_end = max(end_times)
    min_end = min(end_times)
    lag_us = max_end - min_end

    # Normalize by average iteration time
    if iteration_times:
        avg_iter_time = sum(iteration_times) / len(iteration_times)
        normalized_lag = lag_us / avg_iter_time if avg_iter_time > 0 else 0
    else:
        # Normalize by average end time as fallback
        avg_end = sum(end_times) / len(end_times)
        normalized_lag = lag_us / avg_end if avg_end > 0 else 0

    import sys
    print("Straggler Analysis:", file=sys.stderr)
    print(f"  Ranks analyzed: {len(rank_end_times)}", file=sys.stderr)
    max_rank = max(rank_end_times.keys(), key=lambda k: rank_end_times[k])
    min_rank = min(rank_end_times.keys(), key=lambda k: rank_end_times[k])
    print(f"  Max end time: {max_end:.2f} us (rank {max_rank})", file=sys.stderr)
    print(f"  Min end time: {min_end:.2f} us (rank {min_rank})", file=sys.stderr)
    print(f"  Lag: {lag_us:.2f} us", file=sys.stderr)
    print(f"  Normalized lag: {normalized_lag:.4f}", file=sys.stderr)

    return normalized_lag


def _find_rank_traces(directory: str) -> dict[int, str]:
    """Find trace files organized by rank."""
    rank_traces = {}
    trace_dir = Path(directory)

    # Look for trace files with various naming patterns
    trace_patterns = [
        "kineto_trace*.json",
        "rank*_trace.json",
        "*trace*.json",
    ]

    trace_files = []
    for pattern in trace_patterns:
        trace_files.extend(trace_dir.glob(pattern))

    # Also check profile_trace subdirectory
    profile_dir = trace_dir / "profile_trace"
    if profile_dir.exists():
        for pattern in trace_patterns:
            trace_files.extend(profile_dir.glob(pattern))

    for path in trace_files:
        if not path.is_file() or path.suffix != ".json":
            continue

        # Try to extract rank from filename
        name = path.name.replace(".json", "").lower()

        # Try different patterns: rank0_trace, kineto_trace_0, trace_rank_0
        rank = -1
        if name.startswith("rank"):
            # rank0_trace.json -> rank 0
            try:
                rank_part = name.split("_")[0].replace("rank", "")
                rank = int(rank_part)
            except ValueError:
                pass
        elif "rank" in name:
            # kineto_trace_rank_0.json
            parts = name.split("_")
            for i, part in enumerate(parts):
                if part == "rank" and i + 1 < len(parts):
                    try:
                        rank = int(parts[i + 1])
                    except ValueError:
                        pass
                    break
        else:
            # Try last number in filename
            parts = name.replace(".json", "").split("_")
            for part in reversed(parts):
                try:
                    rank = int(part)
                    break
                except ValueError:
                    continue

        if rank >= 0:
            rank_traces[rank] = str(path)

    return rank_traces


def _analyze_rank_trace(trace_path: str) -> tuple[float, float]:
    """Analyze a single rank's trace.

    Returns:
        Tuple[float, float]: (end_time_us, avg_iteration_time_us)
    """
    try:
        with Path(trace_path).open() as f:
            data = json.load(f)

        events = data.get("traceEvents", [])
        if not events:
            return 0.0, 0.0

        # Find the last event end time (not just kernels, as some traces
        # may not have kernel category)
        max_end_time = 0.0
        min_start_time = float("inf")

        for event in events:
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)

            if ts > 0:
                min_start_time = min(min_start_time, ts)
                if dur > 0:
                    max_end_time = max(max_end_time, ts + dur)
                else:
                    max_end_time = max(max_end_time, ts)

        if max_end_time == 0:
            return 0.0, 0.0

        total_time = max_end_time - min_start_time

        # Estimate iteration time (assume 1 iteration per trace file for TorchTitan)
        iter_time = total_time

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {trace_path}: {e}")
        return 0.0, 0.0
    else:
        return max_end_time, iter_time
