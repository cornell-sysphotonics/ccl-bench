"""Calculate compute-communication overlap metrics from torch profile traces."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any, cast


Interval = tuple[float, float]


_LOGGER = logging.getLogger(__name__)

# Communication patterns for kernels
NCCL_PATTERNS = [
    re.compile(r"nccl", re.IGNORECASE),
    re.compile(r"AllGather", re.IGNORECASE),
    re.compile(r"AllReduce", re.IGNORECASE),
    re.compile(r"ReduceScatter", re.IGNORECASE),
    re.compile(r"Broadcast", re.IGNORECASE),
]


def _load_trace_json(path: Path) -> dict[str, Any] | None:
    """Load a torch trace JSON file."""
    try:
        with path.open() as file:
            return cast("dict[str, Any]", json.load(file))
    except Exception as e:
        _LOGGER.warning("Failed to load trace file %s: %s", path, e)
        return None


def _is_communication_kernel(name: str) -> bool:
    """Check if a kernel is a communication kernel."""
    return any(pattern.search(name) for pattern in NCCL_PATTERNS)


def _extract_kernel_intervals(trace_data: dict[str, Any]) -> tuple[list[Interval], list[Interval]]:
    """Extract compute and communication kernel intervals.

    Returns:
        (compute_intervals, comm_intervals) where each interval is (start, end) in microseconds
    """
    if "traceEvents" not in trace_data:
        return [], []

    events = trace_data["traceEvents"]
    compute_intervals: list[Interval] = []
    comm_intervals: list[Interval] = []

    for event in events:
        if not isinstance(event, dict):
            continue

        # Only look at kernel events
        if event.get("cat") != "kernel":
            continue

        if event.get("ph") != "X":
            continue

        name = event.get("name", "")
        ts = float(event.get("ts", 0))
        dur = float(event.get("dur", 0))

        if dur <= 0:
            continue

        interval = (ts, ts + dur)

        if _is_communication_kernel(name):
            comm_intervals.append(interval)
        else:
            compute_intervals.append(interval)

    return compute_intervals, comm_intervals


def _merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping intervals."""
    if not intervals:
        return []

    # Sort by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # Overlapping, merge
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def _calculate_total_duration(intervals: list[Interval]) -> float:
    """Calculate total duration of merged intervals."""
    merged = _merge_intervals(intervals)
    return sum(end - start for start, end in merged)


def _calculate_overlap(intervals_a: list[Interval], intervals_b: list[Interval]) -> float:
    """Calculate the overlap duration between two sets of intervals."""
    if not intervals_a or not intervals_b:
        return 0.0

    merged_a = _merge_intervals(intervals_a)
    merged_b = _merge_intervals(intervals_b)

    overlap_duration = 0.0

    for a_start, a_end in merged_a:
        for b_start, b_end in merged_b:
            # Calculate intersection
            overlap_start = max(a_start, b_start)
            overlap_end = min(a_end, b_end)

            if overlap_start < overlap_end:
                overlap_duration += overlap_end - overlap_start

    return overlap_duration


def _analyze_rank_overlap(trace_data: dict[str, Any]) -> dict[str, Any]:
    """Analyze compute-communication overlap for a single rank."""
    compute_intervals, comm_intervals = _extract_kernel_intervals(trace_data)

    # Calculate durations
    compute_duration = _calculate_total_duration(compute_intervals)
    comm_duration = _calculate_total_duration(comm_intervals)
    overlap_duration = _calculate_overlap(compute_intervals, comm_intervals)

    # Calculate total trace span
    all_intervals = compute_intervals + comm_intervals
    if all_intervals:
        merged_all = _merge_intervals(all_intervals)
        total_span = merged_all[-1][1] - merged_all[0][0] if merged_all else 0
    else:
        total_span = 0

    # Calculate bubble time (idle time)
    total_kernel_time = _calculate_total_duration(all_intervals)
    bubble_time = total_span - total_kernel_time if total_span > total_kernel_time else 0

    # Calculate overlap efficiency
    overlap_efficiency = overlap_duration / comm_duration if comm_duration > 0 else 0

    return {
        "compute_time_us": compute_duration,
        "compute_time_ms": compute_duration / 1000,
        "communication_time_us": comm_duration,
        "communication_time_ms": comm_duration / 1000,
        "overlap_time_us": overlap_duration,
        "overlap_time_ms": overlap_duration / 1000,
        "bubble_time_us": bubble_time,
        "bubble_time_ms": bubble_time / 1000,
        "total_kernel_time_ms": total_kernel_time / 1000,
        "overlap_efficiency": round(overlap_efficiency * 100, 2),
        "compute_percentage": round(compute_duration / total_span * 100, 2)
        if total_span > 0
        else 0,
        "comm_percentage": round(comm_duration / total_span * 100, 2) if total_span > 0 else 0,
        "bubble_percentage": round(bubble_time / total_span * 100, 2) if total_span > 0 else 0,
        "num_compute_kernels": len(compute_intervals),
        "num_comm_kernels": len(comm_intervals),
    }


def metric_cal(
    trace_dir: str,
    workload_card_path: str | None = None,
    profile_mode: str = "auto",
) -> dict[str, Any]:
    """Calculate compute-communication overlap metrics from profile traces.

    Args:
        trace_dir: Directory containing profile traces
        workload_card_path: Path to workload card YAML file (optional)
        profile_mode: Profile mode ("torch", "nsys", or "auto")

    Returns:
        Dictionary with overlap metrics:
        {
            "overlap_efficiency": float (percentage),
            "compute_time_ms": float,
            "communication_time_ms": float,
            "overlap_time_ms": float,
            "bubble_time_ms": float,
            "per_rank_stats": list,
            "num_ranks": int,
        }
    """
    trace_path = Path(trace_dir)

    if not trace_path.exists():
        return {"error": f"Trace directory does not exist: {trace_dir}"}

    # Find all trace JSON files
    trace_files = list(trace_path.glob("rank*_trace.json"))
    if not trace_files:
        trace_files = list(trace_path.glob("*trace.json"))

    if not trace_files:
        return {"error": f"No torch trace JSON files found in {trace_dir}"}

    _LOGGER.info("Found %s trace files in %s", len(trace_files), trace_dir)

    # Analyze each rank
    per_rank_stats = []

    for trace_file in sorted(trace_files):
        trace_data = _load_trace_json(trace_file)
        if trace_data is None:
            continue

        rank_stats = _analyze_rank_overlap(trace_data)
        rank_stats["rank"] = trace_file.stem
        per_rank_stats.append(rank_stats)

    if not per_rank_stats:
        return {"error": "Could not analyze any trace files"}

    # Calculate aggregate metrics
    avg_compute = sum(r["compute_time_ms"] for r in per_rank_stats) / len(per_rank_stats)
    avg_comm = sum(r["communication_time_ms"] for r in per_rank_stats) / len(per_rank_stats)
    avg_overlap = sum(r["overlap_time_ms"] for r in per_rank_stats) / len(per_rank_stats)
    avg_bubble = sum(r["bubble_time_ms"] for r in per_rank_stats) / len(per_rank_stats)
    avg_efficiency = sum(r["overlap_efficiency"] for r in per_rank_stats) / len(per_rank_stats)

    total_compute_kernels = sum(r["num_compute_kernels"] for r in per_rank_stats)
    total_comm_kernels = sum(r["num_comm_kernels"] for r in per_rank_stats)

    return {
        "avg_overlap_efficiency": round(avg_efficiency, 2),
        "avg_compute_time_ms": round(avg_compute, 2),
        "avg_communication_time_ms": round(avg_comm, 2),
        "avg_overlap_time_ms": round(avg_overlap, 2),
        "avg_bubble_time_ms": round(avg_bubble, 2),
        "total_compute_kernels": total_compute_kernels,
        "total_comm_kernels": total_comm_kernels,
        "per_rank_stats": per_rank_stats,
        "num_ranks": len(per_rank_stats),
    }
