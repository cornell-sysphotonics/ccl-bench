"""Calculate CUDA kernel metrics from torch profile traces."""

from __future__ import annotations

from collections import defaultdict
import json
import logging
from pathlib import Path
import re
from typing import Any


_LOGGER = logging.getLogger(__name__)

# Kernel type classification patterns
KERNEL_TYPES = {
    "gemm": re.compile(r"gemm|gemv|matmul|mm_|_mm|cutlass|cublas", re.IGNORECASE),
    "attention": re.compile(r"flash.*attn|attention|softmax|sdpa", re.IGNORECASE),
    "elementwise": re.compile(
        r"elementwise|ewise|add_|mul_|div_|relu|gelu|silu|sigmoid", re.IGNORECASE
    ),
    "reduction": re.compile(r"reduce|sum|mean|norm|layernorm|rmsnorm", re.IGNORECASE),
    "memory": re.compile(r"copy|memcpy|memset|transpose|permute|contiguous", re.IGNORECASE),
    "nccl": re.compile(r"nccl|AllGather|AllReduce|ReduceScatter|Broadcast", re.IGNORECASE),
    "embedding": re.compile(r"embedding|lookup|gather|scatter|index", re.IGNORECASE),
}


def _load_trace_json(path: Path) -> dict[str, Any] | None:
    """Load a torch trace JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        _LOGGER.warning(f"Failed to load trace file {path}: {e}")
        return None


def _classify_kernel(name: str) -> str:
    """Classify a kernel by type."""
    for kernel_type, pattern in KERNEL_TYPES.items():
        if pattern.search(name):
            return kernel_type
    return "other"


def _analyze_rank_kernels(trace_data: dict[str, Any]) -> dict[str, Any]:
    """Analyze kernel metrics for a single rank."""
    if "traceEvents" not in trace_data:
        return {}

    events = trace_data["traceEvents"]

    # Kernel statistics
    kernel_stats: dict[str, dict] = defaultdict(
        lambda: {
            "count": 0,
            "total_time_us": 0,
            "times": [],
        }
    )

    # Individual kernel tracking (top N by time)
    individual_kernels: list[dict] = []

    # Type-based statistics
    type_stats: dict[str, dict] = defaultdict(
        lambda: {
            "count": 0,
            "total_time_us": 0,
        }
    )

    total_kernel_time = 0.0

    for event in events:
        if not isinstance(event, dict):
            continue

        if event.get("cat") != "kernel":
            continue

        if event.get("ph") != "X":
            continue

        name = event.get("name", "")
        dur = event.get("dur", 0)

        if dur <= 0:
            continue

        # Record individual kernel
        individual_kernels.append(
            {
                "name": name,
                "duration_us": dur,
                "type": _classify_kernel(name),
            }
        )

        # Aggregate by kernel name
        kernel_stats[name]["count"] += 1
        kernel_stats[name]["total_time_us"] += dur
        kernel_stats[name]["times"].append(dur)

        # Aggregate by type
        kernel_type = _classify_kernel(name)
        type_stats[kernel_type]["count"] += 1
        type_stats[kernel_type]["total_time_us"] += dur

        total_kernel_time += dur

    # Get top 10 most time-consuming kernels
    kernel_summaries = []
    for name, stats in sorted(
        kernel_stats.items(), key=lambda x: x[1]["total_time_us"], reverse=True
    )[:10]:
        times = stats["times"]
        kernel_summaries.append(
            {
                "name": name[:80] + "..." if len(name) > 80 else name,
                "count": stats["count"],
                "total_time_ms": stats["total_time_us"] / 1000,
                "avg_time_us": sum(times) / len(times) if times else 0,
                "percentage": round(stats["total_time_us"] / total_kernel_time * 100, 2)
                if total_kernel_time > 0
                else 0,
                "type": _classify_kernel(name),
            }
        )

    # Type breakdown
    type_breakdown = {}
    for kernel_type in sorted(type_stats.keys()):
        stats = type_stats[kernel_type]
        type_breakdown[kernel_type] = {
            "count": stats["count"],
            "total_time_ms": stats["total_time_us"] / 1000,
            "percentage": round(stats["total_time_us"] / total_kernel_time * 100, 2)
            if total_kernel_time > 0
            else 0,
        }

    return {
        "total_kernel_time_ms": total_kernel_time / 1000,
        "num_unique_kernels": len(kernel_stats),
        "total_kernel_launches": sum(s["count"] for s in kernel_stats.values()),
        "top_kernels": kernel_summaries,
        "type_breakdown": type_breakdown,
    }


def metric_cal(
    trace_dir: str,
    workload_card_path: str | None = None,
    profile_mode: str = "auto",
) -> dict[str, Any]:
    """Calculate CUDA kernel metrics from profile traces.

    Args:
        trace_dir: Directory containing profile traces
        workload_card_path: Path to workload card YAML file (optional)
        profile_mode: Profile mode ("torch", "nsys", or "auto")

    Returns:
        Dictionary with kernel metrics:
        {
            "total_kernel_time_ms": float,
            "avg_kernel_launches_per_rank": float,
            "top_kernels": list,
            "type_breakdown": dict,
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

        rank_stats = _analyze_rank_kernels(trace_data)
        if rank_stats:
            rank_stats["rank"] = trace_file.stem
            per_rank_stats.append(rank_stats)

    if not per_rank_stats:
        return {"error": "Could not analyze any trace files"}

    # Calculate aggregate metrics
    def avg_metric(key: str) -> float:
        values = [r.get(key, 0) for r in per_rank_stats]
        return sum(values) / len(values) if values else 0

    # Aggregate type breakdown
    all_types = set()
    for r in per_rank_stats:
        all_types.update(r.get("type_breakdown", {}).keys())

    aggregated_types = {}
    for kernel_type in sorted(all_types):
        counts = [
            r.get("type_breakdown", {}).get(kernel_type, {}).get("count", 0) for r in per_rank_stats
        ]
        times = [
            r.get("type_breakdown", {}).get(kernel_type, {}).get("total_time_ms", 0)
            for r in per_rank_stats
        ]
        pcts = [
            r.get("type_breakdown", {}).get(kernel_type, {}).get("percentage", 0)
            for r in per_rank_stats
        ]

        aggregated_types[kernel_type] = {
            "avg_count": round(sum(counts) / len(counts), 1),
            "avg_time_ms": round(sum(times) / len(times), 2),
            "avg_percentage": round(sum(pcts) / len(pcts), 2),
        }

    # Get top kernels from rank 0 as representative
    top_kernels = per_rank_stats[0].get("top_kernels", []) if per_rank_stats else []

    return {
        "avg_kernel_time_ms": round(avg_metric("total_kernel_time_ms"), 2),
        "avg_unique_kernels": round(avg_metric("num_unique_kernels"), 0),
        "avg_kernel_launches": round(avg_metric("total_kernel_launches"), 0),
        "type_breakdown": aggregated_types,
        "top_kernels": top_kernels,
        "per_rank_stats": per_rank_stats,
        "num_ranks": len(per_rank_stats),
    }
