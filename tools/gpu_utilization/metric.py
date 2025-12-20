"""Calculate GPU utilization metrics from torch profile traces."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


_LOGGER = logging.getLogger(__name__)


def _load_trace_json(path: Path) -> dict[str, Any] | None:
    """Load a torch trace JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        _LOGGER.warning(f"Failed to load trace file {path}: {e}")
        return None


def _merge_intervals(intervals: list[tuple]) -> list[tuple]:
    """Merge overlapping intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def _analyze_rank_utilization(trace_data: dict[str, Any]) -> dict[str, Any]:
    """Analyze GPU utilization for a single rank."""
    if "traceEvents" not in trace_data:
        return {}

    events = trace_data["traceEvents"]

    # Collect kernel intervals
    kernel_intervals = []
    memcpy_intervals = []

    # Track total trace span
    all_timestamps = []

    for event in events:
        if not isinstance(event, dict):
            continue

        if event.get("ph") != "X":
            continue

        cat = event.get("cat", "")
        ts = event.get("ts", 0)
        dur = event.get("dur", 0)

        if ts > 0:
            all_timestamps.append(ts)
            if dur > 0:
                all_timestamps.append(ts + dur)

        if dur <= 0:
            continue

        if cat == "kernel":
            kernel_intervals.append((ts, ts + dur))
        elif cat == "gpu_memcpy":
            memcpy_intervals.append((ts, ts + dur))

    if not all_timestamps:
        return {}

    # Calculate total trace span (GPU activity window)
    trace_start = min(all_timestamps)
    trace_end = max(all_timestamps)
    total_span = trace_end - trace_start

    # Merge intervals to calculate actual busy time
    merged_kernel = _merge_intervals(kernel_intervals)
    merged_memcpy = _merge_intervals(memcpy_intervals)
    merged_all = _merge_intervals(kernel_intervals + memcpy_intervals)

    kernel_busy_time = sum(end - start for start, end in merged_kernel)
    memcpy_busy_time = sum(end - start for start, end in merged_memcpy)
    total_busy_time = sum(end - start for start, end in merged_all)

    idle_time = total_span - total_busy_time

    # Calculate utilization percentages
    kernel_utilization = (kernel_busy_time / total_span * 100) if total_span > 0 else 0
    memcpy_utilization = (memcpy_busy_time / total_span * 100) if total_span > 0 else 0
    total_utilization = (total_busy_time / total_span * 100) if total_span > 0 else 0
    idle_percentage = (idle_time / total_span * 100) if total_span > 0 else 0

    return {
        "total_span_ms": total_span / 1000,
        "kernel_time_ms": kernel_busy_time / 1000,
        "memcpy_time_ms": memcpy_busy_time / 1000,
        "total_busy_time_ms": total_busy_time / 1000,
        "idle_time_ms": idle_time / 1000,
        "kernel_utilization_pct": round(kernel_utilization, 2),
        "memcpy_utilization_pct": round(memcpy_utilization, 2),
        "total_utilization_pct": round(total_utilization, 2),
        "idle_percentage": round(idle_percentage, 2),
        "num_kernel_launches": len(kernel_intervals),
        "num_memcpy_ops": len(memcpy_intervals),
    }


def _get_device_properties(trace_data: dict[str, Any]) -> dict[str, Any]:
    """Extract device properties from trace data."""
    device_props = trace_data.get("deviceProperties", [])
    if device_props and isinstance(device_props, list) and len(device_props) > 0:
        prop = device_props[0]
        return {
            "device_name": prop.get("name", "Unknown"),
            "total_memory_gb": prop.get("totalGlobalMem", 0) / (1024**3),
            "num_sms": prop.get("numSms", 0),
            "compute_capability": f"{prop.get('computeMajor', 0)}.{prop.get('computeMinor', 0)}",
        }
    return {}


def metric_cal(
    trace_dir: str,
    workload_card_path: str | None = None,
    profile_mode: str = "auto",
) -> dict[str, Any]:
    """Calculate GPU utilization metrics from profile traces.

    Args:
        trace_dir: Directory containing profile traces
        workload_card_path: Path to workload card YAML file (optional)
        profile_mode: Profile mode ("torch", "nsys", or "auto")

    Returns:
        Dictionary with GPU utilization metrics:
        {
            "avg_gpu_utilization_pct": float,
            "avg_kernel_utilization_pct": float,
            "avg_idle_percentage": float,
            "device_info": dict,
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

    _LOGGER.info(f"Found {len(trace_files)} trace files in {trace_dir}")

    # Analyze each rank
    per_rank_stats = []
    device_info = {}

    for trace_file in sorted(trace_files):
        trace_data = _load_trace_json(trace_file)
        if trace_data is None:
            continue

        # Get device info from first trace
        if not device_info:
            device_info = _get_device_properties(trace_data)

        rank_stats = _analyze_rank_utilization(trace_data)
        if rank_stats:
            rank_stats["rank"] = trace_file.stem
            per_rank_stats.append(rank_stats)

    if not per_rank_stats:
        return {"error": "Could not analyze any trace files"}

    # Calculate aggregate metrics
    def avg_metric(key: str) -> float:
        values = [r.get(key, 0) for r in per_rank_stats]
        return sum(values) / len(values) if values else 0

    return {
        "avg_gpu_utilization_pct": round(avg_metric("total_utilization_pct"), 2),
        "avg_kernel_utilization_pct": round(avg_metric("kernel_utilization_pct"), 2),
        "avg_memcpy_utilization_pct": round(avg_metric("memcpy_utilization_pct"), 2),
        "avg_idle_percentage": round(avg_metric("idle_percentage"), 2),
        "avg_kernel_launches": round(avg_metric("num_kernel_launches"), 0),
        "avg_memcpy_ops": round(avg_metric("num_memcpy_ops"), 0),
        "device_info": device_info,
        "per_rank_stats": per_rank_stats,
        "num_ranks": len(per_rank_stats),
    }
