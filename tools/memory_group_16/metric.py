"""Calculate memory metrics from torch memory snapshots."""

from __future__ import annotations

import logging
from pathlib import Path
import pickle
from typing import Any, cast


_LOGGER = logging.getLogger(__name__)


def _load_memory_snapshot(path: Path) -> dict[str, Any] | None:
    """Load a torch memory snapshot pickle file."""
    try:
        with path.open("rb") as file:
            return cast("dict[str, Any]", pickle.load(file))  # noqa: S301 safe trusted input
    except Exception as e:
        _LOGGER.warning("Failed to load memory snapshot %s: %s", path, e)
        return None


def _analyze_segments(segments: list[dict]) -> dict[str, Any]:
    """Analyze memory segments from a snapshot."""
    if not segments:
        return {
            "total_allocated_bytes": 0,
            "total_active_bytes": 0,
            "total_reserved_bytes": 0,
            "num_segments": 0,
            "num_blocks": 0,
        }

    total_allocated = 0
    total_active = 0
    total_reserved = 0
    total_requested = 0
    num_blocks = 0

    segment_sizes = []
    block_states = {"active_allocated": 0, "active_pending_free": 0, "inactive": 0}

    for segment in segments:
        total_size = segment.get("total_size", 0)
        allocated_size = segment.get("allocated_size", 0)
        active_size = segment.get("active_size", 0)
        requested_size = segment.get("requested_size", 0)

        total_reserved += total_size
        total_allocated += allocated_size
        total_active += active_size
        total_requested += requested_size
        segment_sizes.append(total_size)

        # Count blocks
        blocks = segment.get("blocks", [])
        num_blocks += len(blocks)

        for block in blocks:
            state = block.get("state", "unknown")
            if state in block_states:
                block_states[state] += 1

    return {
        "total_reserved_bytes": total_reserved,
        "total_reserved_gb": total_reserved / (1024**3),
        "total_allocated_bytes": total_allocated,
        "total_allocated_gb": total_allocated / (1024**3),
        "total_active_bytes": total_active,
        "total_active_gb": total_active / (1024**3),
        "total_requested_bytes": total_requested,
        "total_requested_gb": total_requested / (1024**3),
        "num_segments": len(segments),
        "num_blocks": num_blocks,
        "block_states": block_states,
        "avg_segment_size_mb": (sum(segment_sizes) / len(segment_sizes)) / (1024**2)
        if segment_sizes
        else 0,
        "max_segment_size_mb": max(segment_sizes) / (1024**2) if segment_sizes else 0,
        "min_segment_size_mb": min(segment_sizes) / (1024**2) if segment_sizes else 0,
    }


def _calculate_fragmentation(segments: list[dict]) -> dict[str, float]:
    """Calculate memory fragmentation metrics."""
    if not segments:
        return {"fragmentation_ratio": 0.0, "internal_fragmentation": 0.0}

    total_reserved = 0
    total_allocated = 0
    total_requested = 0

    for segment in segments:
        total_reserved += segment.get("total_size", 0)
        total_allocated += segment.get("allocated_size", 0)
        total_requested += segment.get("requested_size", 0)

    # External fragmentation: reserved but not allocated
    external_frag = (total_reserved - total_allocated) / total_reserved if total_reserved > 0 else 0

    # Internal fragmentation: allocated but not requested
    internal_frag = (
        (total_allocated - total_requested) / total_allocated if total_allocated > 0 else 0
    )

    return {
        "external_fragmentation_ratio": round(external_frag, 4),
        "internal_fragmentation_ratio": round(internal_frag, 4),
        "memory_efficiency": round(total_requested / total_reserved, 4)
        if total_reserved > 0
        else 0,
    }


def _analyze_device_traces(device_traces: list) -> dict[str, Any]:
    """Analyze device allocation traces."""
    if not device_traces:
        return {"num_allocations": 0, "num_frees": 0}

    num_allocs = 0
    num_frees = 0
    alloc_sizes = []

    for trace in device_traces:
        if isinstance(trace, dict):
            action = trace.get("action", "")
            size = trace.get("size", 0)

            if "alloc" in action.lower():
                num_allocs += 1
                alloc_sizes.append(size)
            elif "free" in action.lower():
                num_frees += 1

    return {
        "num_allocations": num_allocs,
        "num_frees": num_frees,
        "avg_allocation_size_mb": (sum(alloc_sizes) / len(alloc_sizes)) / (1024**2)
        if alloc_sizes
        else 0,
        "max_allocation_size_mb": max(alloc_sizes) / (1024**2) if alloc_sizes else 0,
    }


def _analyze_rank_memory(snapshot_data: dict[str, Any]) -> dict[str, Any]:
    """Analyze memory for a single rank."""
    segments = snapshot_data.get("segments", [])
    device_traces = snapshot_data.get("device_traces", [])
    allocator_settings = snapshot_data.get("allocator_settings", {})

    segment_analysis = _analyze_segments(segments)
    fragmentation = _calculate_fragmentation(segments)
    trace_analysis = _analyze_device_traces(device_traces)

    return {
        **segment_analysis,
        **fragmentation,
        **trace_analysis,
        "allocator_settings": allocator_settings,
    }


def _find_memory_snapshots(trace_dir: Path) -> list[Path]:
    """Find memory snapshot files in various locations."""
    snapshots: list[Path] = []

    # Check in trace_dir directly
    snapshots.extend(trace_dir.glob("*memory_snapshot.pickle"))

    # Check in parent's memory_snapshot directory
    parent = trace_dir.parent
    if parent.name == "profile_traces":
        memory_dir = parent.parent / "memory_snapshot" / trace_dir.name
        if memory_dir.exists():
            snapshots.extend(memory_dir.glob("*memory_snapshot.pickle"))

    # Check sibling memory_snapshot directory
    memory_dir = trace_dir.parent / "memory_snapshot"
    if memory_dir.exists():
        # Try to match iteration
        iter_dir = memory_dir / trace_dir.name
        if iter_dir.exists():
            snapshots.extend(iter_dir.glob("*memory_snapshot.pickle"))
        else:
            snapshots.extend(memory_dir.glob("*memory_snapshot.pickle"))

    return sorted(set(snapshots))


def metric_cal(
    trace_dir: str,
    workload_card_path: str | None = None,
    profile_mode: str = "auto",
) -> dict[str, Any]:
    """Calculate memory metrics from memory snapshots.

    Args:
        trace_dir: Directory containing profile traces (will look for memory_snapshot nearby)
        workload_card_path: Path to workload card YAML file (optional)
        profile_mode: Profile mode ("torch", "nsys", or "auto")

    Returns:
        Dictionary with memory metrics:
        {
            "peak_memory_gb": float,
            "memory_efficiency": float,
            "fragmentation_ratio": float,
            "per_rank_stats": list,
            "num_ranks": int,
        }
    """
    trace_path = Path(trace_dir)

    if not trace_path.exists():
        return {"error": f"Trace directory does not exist: {trace_dir}"}

    # Find memory snapshot files
    snapshot_files = _find_memory_snapshots(trace_path)

    if not snapshot_files:
        return {"error": f"No memory snapshot files found near {trace_dir}"}

    _LOGGER.info("Found %s memory snapshot files", len(snapshot_files))

    # Analyze each rank
    per_rank_stats = []

    for snapshot_file in snapshot_files:
        snapshot_data = _load_memory_snapshot(snapshot_file)
        if snapshot_data is None:
            continue

        rank_stats = _analyze_rank_memory(snapshot_data)
        rank_stats["rank"] = snapshot_file.stem.replace("_memory_snapshot", "")
        per_rank_stats.append(rank_stats)

    if not per_rank_stats:
        return {"error": "Could not analyze any memory snapshots"}

    # Calculate aggregate metrics
    peak_reserved = max(r["total_reserved_gb"] for r in per_rank_stats)
    peak_allocated = max(r["total_allocated_gb"] for r in per_rank_stats)
    peak_active = max(r["total_active_gb"] for r in per_rank_stats)

    avg_efficiency = sum(r["memory_efficiency"] for r in per_rank_stats) / len(per_rank_stats)
    avg_ext_frag = sum(r["external_fragmentation_ratio"] for r in per_rank_stats) / len(
        per_rank_stats
    )
    avg_int_frag = sum(r["internal_fragmentation_ratio"] for r in per_rank_stats) / len(
        per_rank_stats
    )

    total_segments = sum(r["num_segments"] for r in per_rank_stats)
    total_blocks = sum(r["num_blocks"] for r in per_rank_stats)

    return {
        "peak_reserved_memory_gb": round(peak_reserved, 3),
        "peak_allocated_memory_gb": round(peak_allocated, 3),
        "peak_active_memory_gb": round(peak_active, 3),
        "avg_memory_efficiency": round(avg_efficiency, 4),
        "avg_external_fragmentation": round(avg_ext_frag, 4),
        "avg_internal_fragmentation": round(avg_int_frag, 4),
        "total_segments": total_segments,
        "total_blocks": total_blocks,
        "per_rank_stats": per_rank_stats,
        "num_ranks": len(per_rank_stats),
    }
