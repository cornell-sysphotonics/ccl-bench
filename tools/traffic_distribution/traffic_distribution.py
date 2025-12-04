"""Traffic Distribution metric calculation.

Classifies and measures communication traffic by parallelism type:
- DP (Data Parallel): gradient all-reduce
- TP (Tensor Parallel): activation/weight all-reduce
- PP (Pipeline Parallel): point-to-point send/recv
- EP (Expert Parallel): all-to-all for MoE

NVTX Dependency: Partial
- mode='nvtx': Uses NVTX ranges for accurate classification (best accuracy)
- mode='heuristic': Uses kernel name patterns and tensor size heuristics (less accurate)
- mode='auto': Uses NVTX if available, falls back to heuristic

Without NVTX, classification relies on:
- PP: SendRecv kernels
- EP: AllToAll kernels
- DP/TP: Heuristic based on kernel patterns (may be inaccurate)
"""

from __future__ import annotations

from collections import defaultdict
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


# Classification modes
MODE_NVTX = "nvtx"
MODE_HEURISTIC = "heuristic"
MODE_AUTO = "auto"


def metric_cal(directory: str, profile_mode: str = "auto") -> dict[str, Any]:
    """Calculate traffic distribution by parallelism type.

    Args:
        directory: Path to the trace directory containing trace files.
        profile_mode: Not used directly, but kept for API consistency.

    Returns:
        Dictionary with traffic distribution metrics:
            - dp_bytes: Data parallel traffic in bytes (estimated from duration)
            - tp_bytes: Tensor parallel traffic in bytes
            - pp_bytes: Pipeline parallel traffic in bytes
            - ep_bytes: Expert parallel traffic in bytes
            - unknown_bytes: Unclassified traffic
            - total_bytes: Total communication traffic
            - fractions: Dict with fraction of each type
            - total_comm_calls: Total number of communication calls
            - mode: Classification mode used ('nvtx' or 'heuristic')
    """
    traffic = _analyze_traffic_distribution(directory)

    # Calculate totals
    dp_bytes = traffic.get("DP", 0.0)
    tp_bytes = traffic.get("TP", 0.0)
    pp_bytes = traffic.get("PP", 0.0)
    ep_bytes = traffic.get("EP", 0.0)
    unknown_bytes = traffic.get("unknown", 0.0)
    total_bytes = dp_bytes + tp_bytes + pp_bytes + ep_bytes + unknown_bytes

    # Calculate fractions
    fractions = {}
    if total_bytes > 0:
        fractions = {
            "dp": dp_bytes / total_bytes,
            "tp": tp_bytes / total_bytes,
            "pp": pp_bytes / total_bytes,
            "ep": ep_bytes / total_bytes,
            "unknown": unknown_bytes / total_bytes,
        }
    else:
        fractions = {"dp": 0.0, "tp": 0.0, "pp": 0.0, "ep": 0.0, "unknown": 0.0}

    result = {
        "dp_bytes": dp_bytes,
        "tp_bytes": tp_bytes,
        "pp_bytes": pp_bytes,
        "ep_bytes": ep_bytes,
        "unknown_bytes": unknown_bytes,
        "total_bytes": total_bytes,
        "fractions": fractions,
        "total_comm_calls": int(traffic.get("total_calls", 0)),
        "mode": traffic.get("mode", "heuristic"),
    }

    # Print summary
    mode = result["mode"]
    print(f"Traffic Distribution Analysis (mode: {mode}):", file=sys.stderr)
    print(f"  DP: {dp_bytes:.2f} bytes ({fractions['dp'] * 100:.1f}%)", file=sys.stderr)
    print(f"  TP: {tp_bytes:.2f} bytes ({fractions['tp'] * 100:.1f}%)", file=sys.stderr)
    print(f"  PP: {pp_bytes:.2f} bytes ({fractions['pp'] * 100:.1f}%)", file=sys.stderr)
    print(f"  EP: {ep_bytes:.2f} bytes ({fractions['ep'] * 100:.1f}%)", file=sys.stderr)
    print(f"  Unknown: {unknown_bytes:.2f} bytes ({fractions['unknown'] * 100:.1f}%)", file=sys.stderr)
    print(f"  Total comm calls: {result['total_comm_calls']}", file=sys.stderr)

    if mode == MODE_HEURISTIC:
        print("  Warning: Using heuristic mode. DP/TP classification may be inaccurate.", file=sys.stderr)
        print("           For accurate classification, use NVTX ranges to tag parallel types.", file=sys.stderr)

    return result


def _analyze_traffic_distribution(directory: str) -> dict[str, Any]:
    """Analyze traffic distribution from trace files."""
    traffic: dict[str, Any] = {}
    traffic["total_calls"] = 0
    traffic["mode"] = MODE_HEURISTIC  # Default

    trace_dir = Path(directory)

    # Look for trace files
    trace_patterns = [
        "kineto_trace*.json",
        "rank*_trace.json",
        "*trace*.json",
    ]

    trace_files: set[Path] = set()  # Use set to avoid duplicates
    for pattern in trace_patterns:
        trace_files.update(trace_dir.glob(pattern))

    # Also check profile_trace subdirectory
    profile_dir = trace_dir / "profile_trace"
    if profile_dir.exists():
        for pattern in trace_patterns:
            trace_files.update(profile_dir.glob(pattern))

    for path in trace_files:
        if path.is_file() and path.suffix == ".json":
            trace_traffic = _analyze_single_trace(path)
            for key, value in trace_traffic.items():
                if key == "mode":
                    # If any trace has NVTX, use that mode
                    if value == MODE_NVTX:
                        traffic["mode"] = MODE_NVTX
                elif key == "total_calls":
                    traffic["total_calls"] = traffic.get("total_calls", 0) + value
                else:
                    traffic[key] = traffic.get(key, 0.0) + value

    return traffic


def _analyze_single_trace(trace_path: Path) -> dict[str, Any]:
    """Analyze a single trace file for traffic distribution."""
    traffic: dict[str, Any] = {}
    mode = MODE_HEURISTIC

    try:
        with trace_path.open() as f:
            data = json.load(f)

        events = data.get("traceEvents", [])

        # Build NVTX range context for classification
        nvtx_ranges = _extract_nvtx_ranges(events)
        if nvtx_ranges:
            mode = MODE_NVTX

        # First pass: find nccl:coalesced events and determine their type
        # based on what c10d operations they contain
        coalesced_events = []
        c10d_events = []

        for event in events:
            name = event.get("name", "")
            cat = event.get("cat", "")
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)

            if name == "nccl:coalesced" and cat == "user_annotation" and dur > 0:
                coalesced_events.append((ts, ts + dur, dur))
            elif cat == "cpu_op" and name.startswith("c10d::") and ts > 0:
                c10d_events.append((name, ts))

        # Build coalesced -> classification map
        coalesced_classification: dict[tuple[float, float], str] = {}
        for c_start, c_end, c_dur in coalesced_events:
            # Find c10d operations inside this coalesced range
            ops_inside = [name for name, ts in c10d_events if c_start <= ts <= c_end]

            # Classify based on operations inside
            has_send_recv = any("send" in op.lower() or "recv" in op.lower() for op in ops_inside)
            has_allreduce = any("allreduce" in op.lower() for op in ops_inside)
            has_allgather = any("allgather" in op.lower() for op in ops_inside)
            has_reducescatter = any("reduce_scatter" in op.lower() for op in ops_inside)
            has_alltoall = any("alltoall" in op.lower() for op in ops_inside)

            # Priority: if it has send/recv, it's PP (pipeline parallelism)
            if has_send_recv and not has_allreduce and not has_allgather:
                coalesced_classification[(c_start, c_end)] = "PP"
            elif has_alltoall:
                coalesced_classification[(c_start, c_end)] = "EP"
            elif has_reducescatter:
                coalesced_classification[(c_start, c_end)] = "DP"
            elif has_allreduce:
                coalesced_classification[(c_start, c_end)] = "unknown"  # Could be DP or TP
            else:
                coalesced_classification[(c_start, c_end)] = "unknown"

        # Second pass: classify each communication event
        for event in events:
            cat = event.get("cat", "")
            name = event.get("name", "")
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)

            if not _is_comm_kernel(name, cat):
                continue

            traffic["total_calls"] = traffic.get("total_calls", 0) + 1

            # Special handling for nccl:coalesced - use pre-computed classification
            if name == "nccl:coalesced":
                category = coalesced_classification.get((ts, ts + dur), "unknown")
            else:
                # Classify by NVTX context or kernel name patterns
                category = _classify_comm_op(name, ts, dur, nvtx_ranges, mode)

            # Use duration as a proxy for bytes (actual bytes would need args parsing)
            # This is a simplification - in reality we'd need to parse kernel args
            traffic[category] = traffic.get(category, 0.0) + dur

        traffic["mode"] = mode

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {trace_path}: {e}", file=sys.stderr)

    return traffic


def _extract_nvtx_ranges(events: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    """Extract NVTX ranges that might indicate parallelism type.

    Only includes ranges that contain parallelism-related keywords.
    Generic user_annotations (like ProfilerStep, Optimizer, etc.) are excluded.
    """
    ranges: list[tuple[float, float, str]] = []

    # Keywords that indicate parallelism markers
    parallelism_keywords = [
        "dp", "tp", "pp", "ep",
        "data_parallel", "tensor_parallel", "pipeline", "expert",
        "fsdp", "ddp", "moe", "gradient_sync", "all_reduce_grad",
    ]

    for event in events:
        # NVTX events have category "nvtx" or similar
        cat = event.get("cat", "").lower()
        if "nvtx" not in cat and cat != "user_annotation":
            continue

        name = event.get("name", "").lower()
        ts = event.get("ts", 0)
        dur = event.get("dur", 0)

        # Only include if it looks like a parallelism marker
        if ts > 0 and any(kw in name for kw in parallelism_keywords):
            ranges.append((ts, ts + dur, name))

    return ranges


def _is_comm_kernel(name: str, cat: str) -> bool:
    """Check if an event is a communication kernel or operation."""
    # Accept kernel, cpu_op, and user_annotation categories
    if cat not in ("kernel", "cpu_op", "user_annotation"):
        return False

    # Check for operator-level NCCL events
    if name.startswith(_NCCL_OPERATOR_NAMES) or name in _NCCL_OPERATOR_NAMES:
        return True

    comm_patterns = [
        "nccl",
        "allreduce",
        "reducescatter",
        "allgather",
        "broadcast",
        "reduce",
        "sendrecv",
        "alltoall",
        "p2p",
        "send",
        "recv",
    ]

    name_lower = name.lower()
    return any(p in name_lower for p in comm_patterns)


def _classify_comm_op(
    name: str,
    ts: float,
    dur: float,
    nvtx_ranges: list[tuple[float, float, str]],
    mode: str,
) -> str:
    """Classify a communication operation by parallelism type.

    Uses:
    1. NVTX range context if available (mode='nvtx')
    2. Kernel name patterns (mode='heuristic')
    """
    name_lower = name.lower()

    # Mode: NVTX - Check NVTX context first for accurate classification
    # Only use NVTX if we find actual parallelism markers (not just any user_annotation)
    if mode == MODE_NVTX and nvtx_ranges:
        for range_start, range_end, range_name in nvtx_ranges:
            if range_start <= ts <= range_end:
                # Check for explicit parallelism markers
                # Use word boundaries or explicit patterns to avoid false matches
                # (e.g., "ProfilerStep" should not match "ep")
                range_name_parts = range_name.replace("_", " ").replace(":", " ").split()

                if any(p in ["dp", "data_parallel", "gradient"] for p in range_name_parts):
                    return "DP"
                if any(p in ["tp", "tensor_parallel"] for p in range_name_parts):
                    return "TP"
                if any(p in ["pp", "pipeline"] for p in range_name_parts):
                    return "PP"
                if any(p in ["ep", "expert", "moe"] for p in range_name_parts):
                    return "EP"

                # Also check for underscore-separated patterns
                if "data_parallel" in range_name or "_dp_" in range_name or range_name.startswith("dp_") or range_name.endswith("_dp"):
                    return "DP"
                if "tensor_parallel" in range_name or "_tp_" in range_name or range_name.startswith("tp_") or range_name.endswith("_tp"):
                    return "TP"
                if "pipeline_parallel" in range_name or "_pp_" in range_name or range_name.startswith("pp_") or range_name.endswith("_pp"):
                    return "PP"
                if "expert_parallel" in range_name or "_ep_" in range_name or range_name.startswith("ep_") or range_name.endswith("_ep"):
                    return "EP"

    # Heuristic classification based on kernel name patterns

    # PP: Point-to-point operations (SendRecv, send, recv)
    # These are definitively PP since they're used for pipeline stage communication
    if "sendrecv" in name_lower:
        return "PP"
    # c10d::send and c10d::recv_ are PP operations
    if "c10d::send" in name_lower or "c10d::recv" in name_lower:
        return "PP"
    if "nccl:send" in name_lower or "nccl:recv" in name_lower:
        return "PP"
    if name_lower.endswith("_send") or name_lower.endswith("_recv"):
        return "PP"
    if "nccldevkernel_sendrecv" in name_lower or "ncclkernel_sendrecv" in name_lower:
        return "PP"

    # EP: All-to-all operations (typically used in MoE expert parallelism)
    # All-to-all is strongly associated with expert parallelism
    if "alltoall" in name_lower or "all_to_all" in name_lower:
        return "EP"

    # TP vs DP: Both use AllReduce/ReduceScatter/AllGather
    # Heuristic: Without NVTX, we can try to distinguish by operation type
    # - ReduceScatter + AllGather often used in FSDP/ZeRO (DP)
    # - AllReduce used in both TP and traditional DP

    # ReduceScatter is often used in FSDP (DP) for gradient sharding
    if "reducescatter" in name_lower or "reduce_scatter" in name_lower:
        # This is commonly used in DP (FSDP/ZeRO)
        return "DP"

    # AllGather is often used in FSDP (DP) or TP for weight collection
    if "allgather" in name_lower or "all_gather" in name_lower:
        # This is ambiguous - used in both DP (FSDP) and TP
        # Default to DP as it's more common in FSDP setups
        return "unknown"

    # AllReduce is used in both TP and DP
    # Without additional context, classify as unknown
    if "allreduce" in name_lower or "all_reduce" in name_lower:
        return "unknown"

    # Broadcast is less common, could be either
    if "broadcast" in name_lower:
        return "unknown"

    # nccl:coalesced wraps multiple operations, mark as unknown
    if "coalesced" in name_lower:
        return "unknown"

    return "unknown"
