"""Iteration Wall-Clock Time metric calculation.

Measures the average wall-clock time for a single training iteration
from profiling traces.
"""

import json
from pathlib import Path


def metric_cal(directory: str) -> float:
    """Calculate average iteration wall-clock time from trace data.

    Args:
        directory (str): Path to the trace directory containing trace files.

    Returns:
        float: Average iteration time in milliseconds.
    """
    # Try to find iteration markers in traces
    iter_times = _find_iteration_times(directory)

    if not iter_times:
        # Fallback: estimate from total trace duration and assumed iteration count
        print("Warning: No iteration markers found, estimating from total duration")
        return _estimate_iter_time(directory)

    # Return average iteration time in milliseconds
    return sum(iter_times) / len(iter_times)


def _find_iteration_times(directory: str) -> list[float]:
    """Find iteration times from NVTX ranges or other markers in traces.

    Returns:
        List[float]: List of iteration times in milliseconds.
    """
    iter_times = []
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

    # Deduplicate
    trace_files = list(set(trace_files))

    for path in trace_files:
        if path.is_file() and path.suffix == ".json":
            times = _extract_iter_times_from_kineto(path)
            if times:
                iter_times.extend(times)

    return iter_times


def _extract_iter_times_from_kineto(trace_path: Path) -> list[float]:
    """Extract iteration times from Kineto trace NVTX ranges."""
    iter_times = []

    try:
        with trace_path.open() as f:
            data = json.load(f)

        events = data.get("traceEvents", [])

        # Look for iteration markers (NVTX ranges named "iteration" or "step")
        iteration_markers = []
        for event in events:
            name = event.get("name", "").lower()
            if "iteration" in name or "step" in name or "train_step" in name:
                ts = event.get("ts", 0)
                dur = event.get("dur", 0)
                if ts > 0 and dur > 0:
                    iteration_markers.append((ts, dur))

        # Convert to milliseconds
        for _, dur in iteration_markers:
            iter_times.append(dur / 1000.0)  # microseconds to milliseconds

        # If no explicit markers, try to identify iteration boundaries from
        # ProfilerStep markers
        if not iter_times:
            profiler_steps = []
            for event in events:
                name = event.get("name", "")
                if "ProfilerStep" in name:
                    ts = event.get("ts", 0)
                    dur = event.get("dur", 0)
                    if ts > 0:
                        profiler_steps.append((ts, dur))

            # Calculate time between profiler steps
            if len(profiler_steps) > 1:
                profiler_steps.sort(key=lambda x: x[0])
                for i in range(len(profiler_steps) - 1):
                    time_diff = profiler_steps[i + 1][0] - profiler_steps[i][0]
                    iter_times.append(time_diff / 1000.0)  # microseconds to ms

    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Error reading {trace_path}: {e}")

    return iter_times


def _estimate_iter_time(directory: str) -> float:
    """Estimate iteration time from total trace duration.

    Assumes 5 iterations if not specified.
    """
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
        try:
            with path.open() as f:
                data = json.load(f)

            events = data.get("traceEvents", [])
            if not events:
                continue

            min_ts = float("inf")
            max_ts = float("-inf")

            for event in events:
                ts = event.get("ts", 0)
                dur = event.get("dur", 0)
                if ts > 0:
                    min_ts = min(min_ts, ts)
                    max_ts = max(max_ts, ts + dur)

            if min_ts < float("inf") and max_ts > float("-inf"):
                total_time_us = max_ts - min_ts
                # Assume 1 iteration per trace file (TorchTitan saves per-iteration)
                return total_time_us / 1000.0  # Convert to ms

        except Exception as e:
            print(f"Error reading {path}: {e}")

    return 0.0
