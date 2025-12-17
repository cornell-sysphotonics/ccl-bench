"""Calculate FSDP-specific metrics from torch profile traces."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

# FSDP operation patterns
FSDP_OPS = {
    "all_gather": re.compile(r"FSDP::all_gather\s*\(([^)]+)\)"),
    "reduce_scatter": re.compile(r"FSDP::reduce_scatter|FSDP::post_backward_reduce"),
    "reshard": re.compile(r"FSDP::.*reshard|FSDP::post_backward_reshard"),
    "prefetch": re.compile(r"FSDP::.*prefetch"),
    "copy_in": re.compile(r"fsdp::all_gather_copy_in|FSDP::all_gather_copy_in"),
    "copy_out": re.compile(r"FSDP::all_gather_copy_out"),
    "pre_forward": re.compile(r"FSDP::pre_forward"),
    "post_forward": re.compile(r"FSDP::post_forward"),
    "pre_backward": re.compile(r"FSDP::pre_backward"),
    "post_backward": re.compile(r"FSDP::post_backward(?!_reduce|_reshard|_accumulate)"),
    "accumulate": re.compile(r"FSDP::post_backward_accumulate"),
}


def _load_trace_json(path: Path) -> dict[str, Any] | None:
    """Load a torch trace JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        _LOGGER.warning(f"Failed to load trace file {path}: {e}")
        return None


def _extract_layer_from_name(name: str) -> str | None:
    """Extract layer identifier from FSDP operation name."""
    # Match patterns like "FSDP::all_gather (layers.26)" or "(norm, output)"
    match = re.search(r"\(([^)]+)\)", name)
    if match:
        return match.group(1).strip()
    return None


def _classify_fsdp_event(name: str) -> str | None:
    """Classify an FSDP event by operation type."""
    for op_type, pattern in FSDP_OPS.items():
        if pattern.search(name):
            return op_type
    
    if "FSDP::" in name or "fsdp::" in name:
        return "other_fsdp"
    
    return None


def _analyze_rank_fsdp(trace_data: dict[str, Any]) -> dict[str, Any]:
    """Analyze FSDP operations for a single rank."""
    if "traceEvents" not in trace_data:
        return {}
    
    events = trace_data["traceEvents"]
    
    # Operation statistics
    op_stats: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "total_time_us": 0,
        "times": [],
    })
    
    # Per-layer statistics
    layer_stats: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "total_time_us": 0,
    })
    
    # Track total time and FSDP time
    total_fsdp_time = 0.0
    total_trace_time = 0.0
    
    # Get trace time span
    timestamps = []
    
    for event in events:
        if not isinstance(event, dict):
            continue
        
        ts = event.get("ts")
        dur = event.get("dur", 0)
        
        if ts is not None:
            timestamps.append(ts)
            if dur > 0:
                timestamps.append(ts + dur)
        
        # Only look at duration events
        if event.get("ph") != "X":
            continue
        
        cat = event.get("cat", "")
        if cat not in ["user_annotation", "cpu_op"]:
            continue
        
        name = event.get("name", "")
        
        # Check if it's an FSDP operation
        op_type = _classify_fsdp_event(name)
        if op_type is None:
            continue
        
        # Record statistics
        op_stats[op_type]["count"] += 1
        op_stats[op_type]["total_time_us"] += dur
        op_stats[op_type]["times"].append(dur)
        total_fsdp_time += dur
        
        # Extract and record layer
        layer = _extract_layer_from_name(name)
        if layer:
            layer_stats[layer]["count"] += 1
            layer_stats[layer]["total_time_us"] += dur
    
    if timestamps:
        total_trace_time = max(timestamps) - min(timestamps)
    
    # Calculate operation summaries
    op_summaries = {}
    for op_type, stats in op_stats.items():
        times = stats["times"]
        op_summaries[op_type] = {
            "count": stats["count"],
            "total_time_ms": stats["total_time_us"] / 1000,
            "avg_time_us": sum(times) / len(times) if times else 0,
            "min_time_us": min(times) if times else 0,
            "max_time_us": max(times) if times else 0,
            "percentage_of_fsdp": round(stats["total_time_us"] / total_fsdp_time * 100, 2) if total_fsdp_time > 0 else 0,
        }
    
    # Calculate layer summaries (sorted by layer number)
    def layer_sort_key(layer_name: str) -> tuple:
        match = re.search(r"layers\.(\d+)", layer_name)
        if match:
            return (0, int(match.group(1)))
        return (1, layer_name)
    
    layer_summaries = {}
    for layer in sorted(layer_stats.keys(), key=layer_sort_key):
        stats = layer_stats[layer]
        layer_summaries[layer] = {
            "count": stats["count"],
            "total_time_ms": stats["total_time_us"] / 1000,
        }
    
    # Calculate FSDP overhead
    fsdp_overhead_pct = (total_fsdp_time / total_trace_time * 100) if total_trace_time > 0 else 0
    
    # Calculate communication vs copy overhead
    comm_time = op_stats["all_gather"]["total_time_us"] + op_stats["reduce_scatter"]["total_time_us"]
    copy_time = op_stats["copy_in"]["total_time_us"] + op_stats["copy_out"]["total_time_us"]
    
    return {
        "total_fsdp_time_ms": total_fsdp_time / 1000,
        "total_trace_time_ms": total_trace_time / 1000,
        "fsdp_overhead_percentage": round(fsdp_overhead_pct, 2),
        "communication_time_ms": comm_time / 1000,
        "copy_overhead_ms": copy_time / 1000,
        "operation_breakdown": op_summaries,
        "layer_breakdown": layer_summaries,
        "num_layers": len(layer_summaries),
    }


def metric_cal(
    trace_dir: str,
    workload_card_path: str | None = None,
    profile_mode: str = "auto",
) -> dict[str, Any]:
    """
    Calculate FSDP-specific metrics from profile traces.
    
    Args:
        trace_dir: Directory containing profile traces
        workload_card_path: Path to workload card YAML file (optional)
        profile_mode: Profile mode ("torch", "nsys", or "auto")
    
    Returns:
        Dictionary with FSDP metrics:
        {
            "fsdp_overhead_percentage": float,
            "all_gather_time_ms": float,
            "reduce_scatter_time_ms": float,
            "copy_overhead_ms": float,
            "operation_breakdown": dict,
            "layer_breakdown": dict,
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
    
    for trace_file in sorted(trace_files):
        trace_data = _load_trace_json(trace_file)
        if trace_data is None:
            continue
        
        rank_stats = _analyze_rank_fsdp(trace_data)
        if rank_stats:
            rank_stats["rank"] = trace_file.stem
            per_rank_stats.append(rank_stats)
    
    if not per_rank_stats:
        return {"error": "Could not analyze any trace files"}
    
    # Calculate aggregate metrics
    def avg_metric(key: str) -> float:
        values = [r.get(key, 0) for r in per_rank_stats]
        return sum(values) / len(values) if values else 0
    
    # Aggregate operation breakdowns
    all_ops = set()
    for r in per_rank_stats:
        all_ops.update(r.get("operation_breakdown", {}).keys())
    
    aggregated_ops = {}
    for op in sorted(all_ops):
        counts = [r.get("operation_breakdown", {}).get(op, {}).get("count", 0) for r in per_rank_stats]
        times = [r.get("operation_breakdown", {}).get(op, {}).get("total_time_ms", 0) for r in per_rank_stats]
        aggregated_ops[op] = {
            "avg_count": round(sum(counts) / len(counts), 1),
            "avg_time_ms": round(sum(times) / len(times), 2),
            "min_time_ms": round(min(times), 2),
            "max_time_ms": round(max(times), 2),
        }
    
    return {
        "avg_fsdp_overhead_percentage": round(avg_metric("fsdp_overhead_percentage"), 2),
        "avg_fsdp_time_ms": round(avg_metric("total_fsdp_time_ms"), 2),
        "avg_communication_time_ms": round(avg_metric("communication_time_ms"), 2),
        "avg_copy_overhead_ms": round(avg_metric("copy_overhead_ms"), 2),
        "operation_breakdown": aggregated_ops,
        "per_rank_stats": per_rank_stats,
        "num_ranks": len(per_rank_stats),
    }

