"""Pipeline Bubble metric calculation.

For pipeline parallel training, measures the idle time (bubble) between
microbatches at each pipeline stage.

Bubble ratio = sum_idle_time / total_iteration_time

Lower bubble ratio indicates better pipeline efficiency.

NVTX Dependency: High (for accurate per-stage analysis)
Without NVTX ranges (like PP_STAGE_0, PP_STAGE_1, etc.), this metric
uses a heuristic approach based on gaps between GPU kernel executions.
This provides a rough estimate but may be inaccurate without proper
stage-level tagging.

For accurate pipeline bubble measurement, consider:
1. Using NVTX ranges to tag pipeline stages
2. Using Torch Profiler with per-stage module naming
3. Using TorchTitan-specific tagging if available
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# NCCL operator-level event names (from TorchTitan/PyTorch profiler traces)
_NCCL_OPERATOR_NAMES = (
    "nccl:all_reduce",
    "nccl:reduce_scatter",
    "nccl:all_gather",
    "nccl:broadcast",
    "nccl:reduce",
    "nccl:send",
    "nccl:recv",
    "nccl:coalesced",
    "nccl:all_to_all",
    # c10d distributed operations
    "c10d::allreduce_",
    "c10d::reduce_scatter_",
    "c10d::allgather_",
    "c10d::broadcast_",
    "c10d::reduce_",
    "c10d::send",
    "c10d::recv_",
    "c10d::alltoall",
)


def metric_cal(directory: str, profile_mode: str = "auto") -> dict[str, Any]:
    """Calculate pipeline bubble ratio from trace data.

    Args:
        directory: Path to the trace directory containing trace files.
        profile_mode: Profile mode - 'torch', 'nsys', or 'auto' (detect).

    Returns:
        Dictionary with pipeline bubble metrics:
            - bubble_ratio: Ratio of idle time to total time (0-1)
            - bubble_time_ms: Total bubble (idle) time in milliseconds
            - total_time_ms: Total iteration time in milliseconds
            - num_ranks: Number of ranks analyzed
            - method: Method used for analysis ('nvtx', 'heuristic', or 'none')
            - warning: Any warnings about analysis accuracy
    """
    # Find Kineto trace files for each rank
    rank_traces = _find_rank_traces(directory)

    if not rank_traces:
        print("Warning: No trace files found for bubble analysis", file=sys.stderr)
        return {
            "bubble_ratio": 0.0,
            "bubble_time_ms": 0.0,
            "total_time_ms": 0.0,
            "num_ranks": 0,
            "method": "none",
            "warning": "No trace files found",
        }

    # Try to calculate bubble ratio using the best available method
    return _calculate_bubble_ratio(rank_traces)


def _find_rank_traces(directory: str) -> dict[int, str]:
    """Find trace files organized by rank."""
    rank_traces: dict[int, str] = {}
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


def _calculate_bubble_ratio(rank_traces: dict[int, str]) -> dict[str, Any]:
    """Calculate bubble ratio from rank traces.

    For pipeline parallelism, bubbles occur when a pipeline stage
    is waiting for activations from the previous stage.

    Without NVTX stage markers, uses heuristic based on kernel gaps.
    """
    all_bubbles: list[float] = []
    total_times: list[float] = []
    method = "heuristic"  # Default to heuristic without NVTX
    has_pp_markers = False

    for trace_path in rank_traces.values():
        result = _analyze_single_trace(trace_path)
        if result["total_time"] > 0:
            all_bubbles.append(result["bubble_time"])
            total_times.append(result["total_time"])
            if result["has_pp_markers"]:
                has_pp_markers = True
                method = "nvtx"

    if not total_times:
        return {
            "bubble_ratio": 0.0,
            "bubble_time_ms": 0.0,
            "total_time_ms": 0.0,
            "num_ranks": len(rank_traces),
            "method": "none",
            "warning": "Could not analyze any trace files",
        }

    # Average bubble ratio across ranks
    total_bubble = sum(all_bubbles)
    total_duration = sum(total_times)
    bubble_ratio = total_bubble / total_duration if total_duration > 0 else 0.0

    # Convert to milliseconds (from microseconds)
    bubble_time_ms = total_bubble / 1000.0
    total_time_ms = total_duration / 1000.0

    # Generate warning if using heuristic
    warning = None
    if method == "heuristic":
        warning = (
            "Using heuristic gap-based analysis. For accurate pipeline bubble "
            "measurement, use NVTX ranges or Torch Profiler with stage tagging."
        )
        print(f"Warning: {warning}", file=sys.stderr)

    print(f"Pipeline bubble analysis ({method} method):", file=sys.stderr)
    print(f"  Ranks analyzed: {len(rank_traces)}", file=sys.stderr)
    print(f"  Total bubble time: {bubble_time_ms:.2f} ms", file=sys.stderr)
    print(f"  Total iteration time: {total_time_ms:.2f} ms", file=sys.stderr)
    print(f"  Bubble ratio: {bubble_ratio:.4f} ({bubble_ratio * 100:.1f}%)", file=sys.stderr)

    return {
        "bubble_ratio": bubble_ratio,
        "bubble_time_ms": bubble_time_ms,
        "total_time_ms": total_time_ms,
        "num_ranks": len(rank_traces),
        "method": method,
        "warning": warning,
    }


def _analyze_single_trace(trace_path: str) -> dict[str, Any]:
    """Analyze a single trace file for pipeline bubbles.

    Returns:
        Dictionary with:
            - bubble_time: Total bubble time in microseconds
            - total_time: Total time in microseconds
            - has_pp_markers: Whether PP markers were found
    """
    try:
        with Path(trace_path).open() as f:
            data = json.load(f)

        events = data.get("traceEvents", [])
        if not events:
            return {"bubble_time": 0.0, "total_time": 0.0, "has_pp_markers": False}

        # Find activity intervals (kernels, cpu_ops, user_annotations)
        activity_intervals: list[tuple[float, float]] = []
        pp_markers: list[tuple[float, float, str]] = []
        profiler_step_duration: float = 0.0
        profiler_step_start: float = 0.0

        # Look for pipeline-related markers
        pp_patterns = ["pp_", "pipeline", "stage", "microbatch", "p2p", "send", "recv"]

        for event in events:
            cat = event.get("cat", "")
            name = event.get("name", "")
            name_lower = name.lower()
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)

            if ts <= 0:
                continue

            # Track ProfilerStep for total iteration time
            if name.startswith("ProfilerStep#") and dur > 0:
                profiler_step_duration = dur
                profiler_step_start = ts

            # Track all activity: kernel events, cpu_ops, and user_annotations with duration
            if dur > 0:
                if cat == "kernel":
                    activity_intervals.append((ts, ts + dur))
                elif cat == "cpu_op":
                    # Include significant compute operations
                    activity_intervals.append((ts, ts + dur))
                elif cat == "user_annotation":
                    # Include communication events from user_annotation
                    if name.startswith(_NCCL_OPERATOR_NAMES) or name in _NCCL_OPERATOR_NAMES:
                        activity_intervals.append((ts, ts + dur))

            # Track pipeline markers (NVTX or similar)
            if any(p in name_lower for p in pp_patterns):
                pp_markers.append((ts, dur, name_lower))

        if not activity_intervals:
            return {"bubble_time": 0.0, "total_time": 0.0, "has_pp_markers": bool(pp_markers)}

        # Sort intervals by start time
        activity_intervals.sort(key=lambda x: x[0])

        # Use ProfilerStep duration if available, otherwise compute from activity
        if profiler_step_duration > 0:
            total_time = profiler_step_duration
            # Merge overlapping intervals to find actual busy time
            merged = _merge_intervals(activity_intervals)
            busy_time = sum(end - start for start, end in merged)
            # Bubble = total iteration time - busy time
            significant_bubble = max(0.0, total_time - busy_time)
        else:
            # Calculate idle time (bubbles) between activities
            min_ts = activity_intervals[0][0]
            max_ts = activity_intervals[-1][1]

            # Merge overlapping intervals to find actual busy time
            merged = _merge_intervals(activity_intervals)

            total_time = max_ts - min_ts

            # Only count significant bubbles (> 1ms gap)
            # This filters out normal scheduling jitter
            significant_bubble = 0.0
            for i in range(len(merged) - 1):
                gap = merged[i + 1][0] - merged[i][1]
                if gap > 1000:  # > 1ms
                    significant_bubble += gap

        return {
            "bubble_time": significant_bubble,
            "total_time": total_time,
            "has_pp_markers": bool(pp_markers),
        }

    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Error reading {trace_path}: {e}", file=sys.stderr)
        return {"bubble_time": 0.0, "total_time": 0.0, "has_pp_markers": False}


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
