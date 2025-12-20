"""Calculate communication metrics from torch profile traces."""

from __future__ import annotations

from collections import defaultdict
import json
import logging
from pathlib import Path
import re
from typing import Any, cast


_LOGGER = logging.getLogger(__name__)

# Communication operation patterns
COMM_PATTERNS = {
    "all_gather": re.compile(r"all_gather|AllGather|allgather", re.IGNORECASE),
    "all_reduce": re.compile(r"all_reduce|AllReduce|allreduce", re.IGNORECASE),
    "reduce_scatter": re.compile(r"reduce_scatter|ReduceScatter|reducescatter", re.IGNORECASE),
    "broadcast": re.compile(r"broadcast|Broadcast", re.IGNORECASE),
    "barrier": re.compile(r"barrier|Barrier", re.IGNORECASE),
    "send": re.compile(r"\bsend\b|Send", re.IGNORECASE),
    "recv": re.compile(r"\brecv\b|Recv", re.IGNORECASE),
}

# NCCL kernel patterns
NCCL_KERNEL_PATTERNS = {
    "all_gather": re.compile(r"ncclKernel.*AllGather|ncclDevKernel.*AllGather", re.IGNORECASE),
    "all_reduce": re.compile(r"ncclKernel.*AllReduce|ncclDevKernel.*AllReduce", re.IGNORECASE),
    "reduce_scatter": re.compile(
        r"ncclKernel.*ReduceScatter|ncclDevKernel.*ReduceScatter", re.IGNORECASE
    ),
    "broadcast": re.compile(r"ncclKernel.*Broadcast|ncclDevKernel.*Broadcast", re.IGNORECASE),
}


def _load_trace_json(path: Path) -> dict[str, Any] | None:
    """Load a torch trace JSON file."""
    try:
        with path.open() as file:
            return cast("dict[str, Any]", json.load(file))
    except Exception as e:
        _LOGGER.warning("Failed to load trace file %s: %s", path, e)
        return None


def _extract_communication_events(trace_data: dict[str, Any]) -> dict[str, list[dict]]:
    """Extract communication-related events from trace data.

    Returns:
        Dictionary mapping operation type to list of events
    """
    if "traceEvents" not in trace_data:
        return {}

    events = trace_data["traceEvents"]
    comm_events: dict[str, list[dict]] = defaultdict(list)

    for event in events:
        if not isinstance(event, dict):
            continue

        # Only look at duration events
        if event.get("ph") != "X":
            continue

        name = event.get("name", "")
        cat = event.get("cat", "")
        dur = event.get("dur", 0)

        # Check user annotations and cpu_ops for high-level communication
        if cat in ["user_annotation", "cpu_op"]:
            for op_type, pattern in COMM_PATTERNS.items():
                if pattern.search(name):
                    comm_events[op_type].append(
                        {
                            "name": name,
                            "duration_us": dur,
                            "category": cat,
                            "ts": event.get("ts", 0),
                        }
                    )
                    break

        # Check CUDA kernels for NCCL operations
        elif cat == "kernel":
            for op_type, pattern in NCCL_KERNEL_PATTERNS.items():
                if pattern.search(name):
                    comm_events[f"{op_type}_kernel"].append(
                        {
                            "name": name,
                            "duration_us": dur,
                            "category": "nccl_kernel",
                            "ts": event.get("ts", 0),
                        }
                    )
                    break

    return dict(comm_events)


def _calculate_total_time(trace_data: dict[str, Any]) -> float:
    """Calculate total trace time in microseconds."""
    if "traceEvents" not in trace_data:
        return 0.0

    events = trace_data["traceEvents"]
    timestamps: list[float] = []

    for event in events:
        if isinstance(event, dict) and event.get("ts") is not None:
            ts = float(event.get("ts", 0))
            dur = float(event.get("dur", 0))
            timestamps.append(ts)
            if dur > 0:
                timestamps.append(ts + dur)

    if not timestamps:
        return 0.0

    return max(timestamps) - min(timestamps)


def _analyze_rank_communication(trace_data: dict[str, Any]) -> dict[str, Any]:
    """Analyze communication for a single rank."""
    comm_events = _extract_communication_events(trace_data)
    total_time_us = _calculate_total_time(trace_data)

    # Calculate per-operation statistics
    op_stats = {}
    total_comm_time_us = 0.0
    total_comm_count = 0

    for op_type, events in comm_events.items():
        if not events:
            continue

        durations = [e["duration_us"] for e in events]
        total_dur = sum(durations)
        count = len(events)

        op_stats[op_type] = {
            "count": count,
            "total_time_us": total_dur,
            "total_time_ms": total_dur / 1000,
            "avg_time_us": total_dur / count if count > 0 else 0,
            "min_time_us": min(durations) if durations else 0,
            "max_time_us": max(durations) if durations else 0,
        }

        # Only count high-level ops (not kernels) for total
        if "_kernel" not in op_type:
            total_comm_time_us += total_dur
            total_comm_count += count

    return {
        "operation_stats": op_stats,
        "total_communication_time_us": total_comm_time_us,
        "total_communication_time_ms": total_comm_time_us / 1000,
        "total_communication_count": total_comm_count,
        "total_trace_time_us": total_time_us,
        "communication_percentage": (total_comm_time_us / total_time_us * 100)
        if total_time_us > 0
        else 0,
    }


def metric_cal(
    trace_dir: str,
    workload_card_path: str | None = None,
    profile_mode: str = "auto",
) -> dict[str, Any]:
    """Calculate communication metrics from profile traces.

    Args:
        trace_dir: Directory containing profile traces
        workload_card_path: Path to workload card YAML file (optional)
        profile_mode: Profile mode ("torch", "nsys", or "auto")

    Returns:
        Dictionary with communication metrics:
        {
            "total_communication_time_ms": float,
            "total_communication_count": int,
            "communication_percentage": float,
            "per_operation_stats": dict,
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
    aggregated_op_stats: dict[str, dict] = defaultdict(
        lambda: {
            "count": 0,
            "total_time_us": 0,
            "times": [],
        }
    )

    for trace_file in sorted(trace_files):
        trace_data = _load_trace_json(trace_file)
        if trace_data is None:
            continue

        rank_stats = _analyze_rank_communication(trace_data)
        rank_stats["rank"] = trace_file.stem
        per_rank_stats.append(rank_stats)

        # Aggregate operation stats
        for op_type, stats in rank_stats.get("operation_stats", {}).items():
            aggregated_op_stats[op_type]["count"] += stats["count"]
            aggregated_op_stats[op_type]["total_time_us"] += stats["total_time_us"]
            aggregated_op_stats[op_type]["times"].append(stats["total_time_us"])

    if not per_rank_stats:
        return {"error": "Could not analyze any trace files"}

    # Calculate aggregate metrics
    avg_comm_time = sum(r["total_communication_time_ms"] for r in per_rank_stats) / len(
        per_rank_stats
    )
    avg_comm_count = sum(r["total_communication_count"] for r in per_rank_stats) / len(
        per_rank_stats
    )
    avg_comm_pct = sum(r["communication_percentage"] for r in per_rank_stats) / len(per_rank_stats)

    # Finalize aggregated stats
    final_op_stats = {}
    for op_type, stats in aggregated_op_stats.items():
        times = stats["times"]
        final_op_stats[op_type] = {
            "total_count": stats["count"],
            "total_time_ms": stats["total_time_us"] / 1000,
            "avg_time_per_rank_ms": (stats["total_time_us"] / len(per_rank_stats)) / 1000
            if per_rank_stats
            else 0,
            "min_rank_time_ms": min(times) / 1000 if times else 0,
            "max_rank_time_ms": max(times) / 1000 if times else 0,
        }

    return {
        "total_communication_time_ms": avg_comm_time,
        "total_communication_count": int(avg_comm_count),
        "communication_percentage": round(avg_comm_pct, 2),
        "per_operation_stats": final_op_stats,
        "per_rank_stats": per_rank_stats,
        "num_ranks": len(per_rank_stats),
    }
