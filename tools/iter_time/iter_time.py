"""Iteration Wall-Clock Time metric calculation.

Measures the average wall-clock time for a single training iteration
from profiling traces.

NVTX Dependency: None (uses ProfilerStep# events instead)
ProfilerStep# events are automatically added by PyTorch Profiler and
don't require NVTX instrumentation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def metric_cal(directory: str, profile_mode: str = "auto") -> dict[str, Any]:
    """Calculate average iteration wall-clock time from trace data.

    Args:
        directory: Path to the trace directory containing trace files.
        profile_mode: Profile mode - 'torch', 'nsys', or 'auto' (detect).

    Returns:
        Dictionary with iteration time metrics:
            - avg_iter_time_ms: Average iteration time in milliseconds
            - min_iter_time_ms: Minimum iteration time
            - max_iter_time_ms: Maximum iteration time
            - std_iter_time_ms: Standard deviation of iteration times
            - num_iterations: Number of iterations detected
            - iter_times_ms: List of individual iteration times
    """
    # Try to find iteration markers in traces
    iter_times = _find_iteration_times(directory, profile_mode)

    if not iter_times:
        print("Warning: No iteration markers found in traces", file=sys.stderr)
        print("  Searched for: ProfilerStep# events", file=sys.stderr)
        return {
            "avg_iter_time_ms": 0.0,
            "min_iter_time_ms": 0.0,
            "max_iter_time_ms": 0.0,
            "std_iter_time_ms": 0.0,
            "num_iterations": 0,
            "iter_times_ms": [],
        }

    # Calculate statistics
    avg_time = sum(iter_times) / len(iter_times)
    min_time = min(iter_times)
    max_time = max(iter_times)

    # Standard deviation
    if len(iter_times) > 1:
        variance = sum((t - avg_time) ** 2 for t in iter_times) / (len(iter_times) - 1)
        std_time = variance ** 0.5
    else:
        std_time = 0.0

    # Sanity check: warn if high variance
    if std_time > avg_time * 0.2 and len(iter_times) > 2:
        print(f"Warning: High iteration time variance detected", file=sys.stderr)
        print(f"  Avg: {avg_time:.2f} ms, Std: {std_time:.2f} ms", file=sys.stderr)
        print(f"  This may indicate profiling warmup or system noise", file=sys.stderr)

    return {
        "avg_iter_time_ms": avg_time,
        "min_iter_time_ms": min_time,
        "max_iter_time_ms": max_time,
        "std_iter_time_ms": std_time,
        "num_iterations": len(iter_times),
        "iter_times_ms": iter_times,
    }


def _find_iteration_times(directory: str, profile_mode: str = "auto") -> list[float]:
    """Find iteration times from ProfilerStep# events in traces.

    ProfilerStep# events are preferred over NVTX ranges because they:
    - Are automatically added by PyTorch Profiler
    - Don't require NVTX instrumentation
    - Are reliable iteration markers

    Returns:
        List of iteration times in milliseconds.
    """
    iter_times: list[float] = []
    trace_dir = Path(directory)

    # Look for trace files with various naming patterns
    trace_patterns = [
        "kineto_trace*.json",
        "rank*_trace.json",
        "*trace*.json",
    ]

    trace_files: list[Path] = []
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
                # Only use times from one trace file (typically rank 0)
                # to avoid double counting
                if not iter_times:
                    iter_times.extend(times)
                break

    return iter_times


def _extract_iter_times_from_kineto(trace_path: Path) -> list[float]:
    """Extract iteration times from Kineto trace using ProfilerStep# events.

    ProfilerStep# events are preferred because they don't require NVTX
    and are automatically added by PyTorch Profiler.
    """
    iter_times: list[float] = []

    try:
        with trace_path.open() as f:
            data = json.load(f)

        events = data.get("traceEvents", [])

        # Primary method: Look for ProfilerStep# events
        # These are X (complete) events with duration
        profiler_steps: list[dict[str, Any]] = []
        for event in events:
            name = event.get("name", "")
            ph = event.get("ph", "")

            # ProfilerStep# events are complete (X) events with name like "ProfilerStep#0"
            if name.startswith("ProfilerStep#") and ph == "X":
                ts = event.get("ts", 0)
                dur = event.get("dur", 0)
                if ts > 0 and dur > 0:
                    profiler_steps.append({"ts": ts, "dur": dur, "name": name})

        if profiler_steps:
            # Sort by timestamp
            profiler_steps.sort(key=lambda e: e["ts"])
            # Use duration of each ProfilerStep as iteration time
            iter_times = [e["dur"] / 1000.0 for e in profiler_steps]  # us -> ms
            print(f"Found {len(iter_times)} ProfilerStep# events in {trace_path.name}", file=sys.stderr)
            return iter_times

        # Fallback: Look for step boundaries using time gaps between steps
        # This handles cases where ProfilerStep duration isn't recorded
        step_starts: list[tuple[int, float]] = []
        for event in events:
            name = event.get("name", "")
            if name.startswith("ProfilerStep#"):
                ts = event.get("ts", 0)
                if ts > 0:
                    # Extract step number
                    try:
                        step_num = int(name.replace("ProfilerStep#", ""))
                        step_starts.append((step_num, ts))
                    except ValueError:
                        pass

        if len(step_starts) > 1:
            step_starts.sort(key=lambda x: x[0])  # Sort by step number
            for i in range(len(step_starts) - 1):
                time_diff = step_starts[i + 1][1] - step_starts[i][1]
                iter_times.append(time_diff / 1000.0)  # us -> ms
            print(f"Computed {len(iter_times)} iteration times from step boundaries", file=sys.stderr)

    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Error reading {trace_path}: {e}", file=sys.stderr)

    return iter_times
