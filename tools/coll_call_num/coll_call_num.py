"""NCCL Communication Call Count metric calculation.

Counts the total number of NCCL collective communication kernel invocations
across all ranks in a distributed training trace.

NVTX Dependency: None
This metric works purely from kernel events and doesn't require NVTX instrumentation.

Supported NCCL operations:
    - AllReduce
    - ReduceScatter
    - AllGather
    - Broadcast
    - Reduce
    - SendRecv (point-to-point)
    - AllToAll
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


# NCCL kernel name prefixes to match (GPU kernel-level events)
# These are the explicit kernel names from NCCL
_NCCL_KERNEL_PREFIXES = (
    "ncclDevKernel_AllReduce",
    "ncclDevKernel_ReduceScatter",
    "ncclDevKernel_AllGather",
    "ncclDevKernel_Broadcast",
    "ncclDevKernel_Reduce",
    "ncclDevKernel_SendRecv",
    "ncclDevKernel_AllToAll",
    # Also match newer NCCL naming patterns
    "ncclKernel_AllReduce",
    "ncclKernel_ReduceScatter",
    "ncclKernel_AllGather",
    "ncclKernel_Broadcast",
    "ncclKernel_Reduce",
    "ncclKernel_SendRecv",
    "ncclKernel_AllToAll",
)

# Operator-level NCCL event names (from TorchTitan/PyTorch profiler traces)
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


def _process_single_trace(
    trace_file: Path,
) -> tuple[int, dict[str, int]] | None:
    """Process a single trace file and count NCCL events.

    Args:
        trace_file: Path to the trace JSON file.

    Returns:
        Tuple of (rank_calls, calls_by_type_delta) or None if processing failed.
    """
    try:
        with trace_file.open() as f:
            trace_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Error decoding JSON in {trace_file}: {e}", file=sys.stderr)
        return None

    rank_calls = 0
    calls_by_type_delta: dict[str, int] = {}

    # Count NCCL events from different trace formats
    for event in trace_data.get("traceEvents", []):
        name = event.get("name", "")
        cat = event.get("cat", "")

        # Format 1: Kernel-level events (cat == "kernel")
        if (
            (cat == "kernel" and name.startswith(_NCCL_KERNEL_PREFIXES))
            or name in _NCCL_OPERATOR_NAMES
            or name.startswith(_NCCL_OPERATOR_NAMES)
        ):
            rank_calls += 1
            # Categorize by type
            coll_type = _categorize_collective(name)
            calls_by_type_delta[coll_type] = calls_by_type_delta.get(coll_type, 0) + 1

    return rank_calls, calls_by_type_delta


def _find_trace_files(directory: Path) -> list[Path]:
    """Find all valid trace JSON files in a directory.

    Searches for trace files using multiple patterns to support different
    naming conventions from TorchTitan, PyTorch Profiler, and Kineto:
        - kineto_trace*.json (legacy naming)
        - rank*_trace.json (rank-based naming)
        - *trace*.json in profile_trace/ subdirectory (TorchTitan default)

    Args:
        directory: Root directory to search for trace files.

    Returns:
        Sorted list of Path objects for found trace files.
    """
    trace_files: list[Path] = []

    # Pattern 1: Direct kineto_trace*.json files in root
    trace_files.extend(directory.glob("kineto_trace*.json"))

    # Pattern 2: rank*_trace.json files
    trace_files.extend(directory.glob("rank*_trace.json"))

    # Pattern 3: TorchTitan's profile_trace subdirectory
    profile_trace_dir = directory / "profile_trace"
    if profile_trace_dir.exists():
        trace_files.extend(profile_trace_dir.glob("*trace*.json"))
        trace_files.extend(profile_trace_dir.glob("*.json"))

    # Pattern 4: Any *trace*.json in root
    trace_files.extend(directory.glob("*trace*.json"))

    # Filter out non-trace files and deduplicate
    valid_traces: list[Path] = []
    seen: set[Path] = set()
    for f in trace_files:
        if f.is_file() and f.suffix == ".json" and f not in seen:
            # Skip obviously wrong files
            if "metadata" in f.name.lower():
                continue
            seen.add(f)
            valid_traces.append(f)

    return sorted(valid_traces)


def metric_cal(directory: str, profile_mode: str = "auto") -> dict[str, Any]:
    """Calculate the total number of NCCL communication calls across all ranks.

    Scans all trace JSON files in the directory (including profile_trace/
    subdirectory) and counts kernel events that match known NCCL collective
    operation patterns.

    Args:
        directory: Path to the directory containing trace JSON files.
        profile_mode: Profile mode - 'torch', 'nsys', or 'auto' (detect).

    Returns:
        Dictionary with communication call metrics:
            - total_calls: Total number of communication calls summed across all ranks
            - calls_per_rank: Dict mapping rank to call count
            - calls_by_type: Dict mapping collective type to count
            - num_traces: Number of trace files processed
    """
    trace_dir = Path(directory)
    if not trace_dir.exists():
        print(f"Warning: Directory not found: {directory}", file=sys.stderr)
        return {
            "total_calls": 0,
            "calls_per_rank": {},
            "calls_by_type": {},
            "num_traces": 0,
        }

    # Find all trace files using flexible pattern matching
    trace_files = _find_trace_files(trace_dir)

    if not trace_files:
        print(f"Warning: No trace JSON files found in {directory}", file=sys.stderr)
        print(
            "  Searched patterns: kineto_trace*.json, rank*_trace.json, profile_trace/*trace*.json",
            file=sys.stderr,
        )
        return {
            "total_calls": 0,
            "calls_per_rank": {},
            "calls_by_type": {},
            "num_traces": 0,
        }

    total_calls = 0
    calls_per_rank: dict[str, int] = {}
    calls_by_type: dict[str, int] = {}
    traces_processed = 0

    for trace_file in trace_files:
        result = _process_single_trace(trace_file)
        if result is None:
            continue

        rank_calls, calls_by_type_delta = result
        total_calls += rank_calls
        traces_processed += 1
        calls_per_rank[trace_file.name] = rank_calls

        # Merge call type counts
        for coll_type, count in calls_by_type_delta.items():
            calls_by_type[coll_type] = calls_by_type.get(coll_type, 0) + count

        print(f"  Processed: {trace_file.name} ({rank_calls} calls)", file=sys.stderr)

    print(f"Processed {traces_processed} trace files", file=sys.stderr)
    print(f"Total communication calls: {total_calls}", file=sys.stderr)

    return {
        "total_calls": total_calls,
        "calls_per_rank": calls_per_rank,
        "calls_by_type": calls_by_type,
        "num_traces": traces_processed,
    }


def _categorize_collective(name: str) -> str:
    """Categorize a collective operation by its type."""
    name_lower = name.lower()

    if "allreduce" in name_lower or "all_reduce" in name_lower:
        return "AllReduce"
    if "reducescatter" in name_lower or "reduce_scatter" in name_lower:
        return "ReduceScatter"
    if "allgather" in name_lower or "all_gather" in name_lower:
        return "AllGather"
    if "alltoall" in name_lower or "all_to_all" in name_lower:
        return "AllToAll"
    if "broadcast" in name_lower:
        return "Broadcast"
    if "sendrecv" in name_lower or "send" in name_lower or "recv" in name_lower:
        return "SendRecv"
    if "reduce" in name_lower:
        return "Reduce"
    return "Other"
