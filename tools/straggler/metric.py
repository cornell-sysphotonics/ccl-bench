"""Calculate straggler detection metrics from torch profile traces."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)


def _load_trace_json(path: Path) -> dict[str, Any] | None:
    """Load a torch trace JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        _LOGGER.warning(f"Failed to load trace file {path}: {e}")
        return None


def _calculate_trace_duration(trace_data: dict[str, Any]) -> float:
    """Calculate total trace duration in microseconds."""
    if "traceEvents" not in trace_data:
        return 0.0
    
    events = trace_data["traceEvents"]
    timestamps = []
    
    for event in events:
        if isinstance(event, dict):
            ts = event.get("ts")
            dur = event.get("dur", 0)
            
            if ts is not None:
                timestamps.append(ts)
                if dur > 0:
                    timestamps.append(ts + dur)
    
    if not timestamps:
        return 0.0
    
    return max(timestamps) - min(timestamps)


def _calculate_kernel_time(trace_data: dict[str, Any]) -> float:
    """Calculate total GPU kernel time in microseconds."""
    if "traceEvents" not in trace_data:
        return 0.0
    
    events = trace_data["traceEvents"]
    total_kernel_time = 0.0
    
    for event in events:
        if not isinstance(event, dict):
            continue
        
        if event.get("cat") == "kernel" and event.get("ph") == "X":
            total_kernel_time += event.get("dur", 0)
    
    return total_kernel_time


def _calculate_communication_time(trace_data: dict[str, Any]) -> float:
    """Calculate total communication time in microseconds."""
    if "traceEvents" not in trace_data:
        return 0.0
    
    events = trace_data["traceEvents"]
    comm_keywords = ["all_gather", "all_reduce", "reduce_scatter", "broadcast", "nccl"]
    total_comm_time = 0.0
    
    for event in events:
        if not isinstance(event, dict):
            continue
        
        if event.get("ph") != "X":
            continue
        
        name = event.get("name", "").lower()
        if any(kw in name for kw in comm_keywords):
            total_comm_time += event.get("dur", 0)
    
    return total_comm_time


def _analyze_rank(trace_data: dict[str, Any]) -> dict[str, float]:
    """Analyze timing metrics for a single rank."""
    total_duration = _calculate_trace_duration(trace_data)
    kernel_time = _calculate_kernel_time(trace_data)
    comm_time = _calculate_communication_time(trace_data)
    
    return {
        "total_duration_ms": total_duration / 1000,
        "kernel_time_ms": kernel_time / 1000,
        "communication_time_ms": comm_time / 1000,
    }


def _calculate_statistics(values: list[float]) -> dict[str, float]:
    """Calculate statistical measures for a list of values."""
    if not values:
        return {
            "mean": 0,
            "std": 0,
            "min": 0,
            "max": 0,
            "range": 0,
            "cv": 0,
        }
    
    n = len(values)
    mean = sum(values) / n
    
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0
    
    min_val = min(values)
    max_val = max(values)
    
    # Coefficient of variation (relative standard deviation)
    cv = (std / mean * 100) if mean > 0 else 0
    
    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "range": round(max_val - min_val, 2),
        "cv": round(cv, 2),
    }


def _detect_stragglers(rank_metrics: list[dict], threshold_std: float = 2.0) -> list[dict]:
    """
    Detect straggler ranks based on duration statistics.
    
    Args:
        rank_metrics: List of per-rank metrics
        threshold_std: Number of standard deviations above mean to consider a straggler
    
    Returns:
        List of straggler information
    """
    if len(rank_metrics) < 2:
        return []
    
    durations = [r["total_duration_ms"] for r in rank_metrics]
    stats = _calculate_statistics(durations)
    
    threshold = stats["mean"] + threshold_std * stats["std"]
    
    stragglers = []
    for r in rank_metrics:
        if r["total_duration_ms"] > threshold:
            deviation = (r["total_duration_ms"] - stats["mean"]) / stats["std"] if stats["std"] > 0 else 0
            stragglers.append({
                "rank": r["rank"],
                "duration_ms": r["total_duration_ms"],
                "deviation_std": round(deviation, 2),
                "slowdown_percentage": round((r["total_duration_ms"] - stats["mean"]) / stats["mean"] * 100, 2),
            })
    
    return sorted(stragglers, key=lambda x: x["deviation_std"], reverse=True)


def metric_cal(
    trace_dir: str,
    workload_card_path: str | None = None,
    profile_mode: str = "auto",
) -> dict[str, Any]:
    """
    Calculate straggler detection metrics from profile traces.
    
    Args:
        trace_dir: Directory containing profile traces
        workload_card_path: Path to workload card YAML file (optional)
        profile_mode: Profile mode ("torch", "nsys", or "auto")
    
    Returns:
        Dictionary with straggler metrics:
        {
            "has_stragglers": bool,
            "num_stragglers": int,
            "stragglers": list,
            "duration_stats": dict,
            "kernel_time_stats": dict,
            "communication_time_stats": dict,
            "slowest_rank": str,
            "fastest_rank": str,
            "load_imbalance_percentage": float,
            "per_rank_metrics": list,
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
    rank_metrics = []
    
    for trace_file in sorted(trace_files):
        trace_data = _load_trace_json(trace_file)
        if trace_data is None:
            continue
        
        metrics = _analyze_rank(trace_data)
        metrics["rank"] = trace_file.stem
        rank_metrics.append(metrics)
    
    if not rank_metrics:
        return {"error": "Could not analyze any trace files"}
    
    # Calculate statistics across ranks
    durations = [r["total_duration_ms"] for r in rank_metrics]
    kernel_times = [r["kernel_time_ms"] for r in rank_metrics]
    comm_times = [r["communication_time_ms"] for r in rank_metrics]
    
    duration_stats = _calculate_statistics(durations)
    kernel_stats = _calculate_statistics(kernel_times)
    comm_stats = _calculate_statistics(comm_times)
    
    # Detect stragglers
    stragglers = _detect_stragglers(rank_metrics)
    
    # Find slowest and fastest ranks
    slowest_rank = max(rank_metrics, key=lambda x: x["total_duration_ms"])
    fastest_rank = min(rank_metrics, key=lambda x: x["total_duration_ms"])
    
    # Calculate load imbalance
    load_imbalance = duration_stats["range"] / duration_stats["mean"] * 100 if duration_stats["mean"] > 0 else 0
    
    # Calculate synchronization overhead (time waiting for slowest rank)
    sync_overhead = sum(duration_stats["max"] - d for d in durations)
    
    return {
        "has_stragglers": len(stragglers) > 0,
        "num_stragglers": len(stragglers),
        "stragglers": stragglers,
        "duration_stats": duration_stats,
        "kernel_time_stats": kernel_stats,
        "communication_time_stats": comm_stats,
        "slowest_rank": {
            "rank": slowest_rank["rank"],
            "duration_ms": slowest_rank["total_duration_ms"],
        },
        "fastest_rank": {
            "rank": fastest_rank["rank"],
            "duration_ms": fastest_rank["total_duration_ms"],
        },
        "load_imbalance_percentage": round(load_imbalance, 2),
        "sync_overhead_ms": round(sync_overhead, 2),
        "coefficient_of_variation": duration_stats["cv"],
        "per_rank_metrics": rank_metrics,
        "num_ranks": len(rank_metrics),
    }

