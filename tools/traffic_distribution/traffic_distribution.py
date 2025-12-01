"""Traffic Distribution metric calculation.

Classifies and measures communication traffic by parallelism type:
- DP (Data Parallel): gradient all-reduce
- TP (Tensor Parallel): activation/weight all-reduce
- PP (Pipeline Parallel): point-to-point send/recv
- EP (Expert Parallel): all-to-all for MoE

Uses NVTX ranges and kernel naming patterns to classify traffic.
"""

from collections import defaultdict
import json
from pathlib import Path


def metric_cal(directory: str) -> dict[str, float]:
    """Calculate traffic distribution by parallelism type.

    Args:
        directory (str): Path to the trace directory containing trace files.

    Returns:
        Dict[str, float]: Dictionary mapping parallelism type to traffic count/duration.
                         Returns as JSON-serializable dict.
    """
    traffic = _analyze_traffic_distribution(directory)

    # Return total duration per category
    result = {
        "DP_traffic_us": traffic.get("DP", 0.0),
        "TP_traffic_us": traffic.get("TP", 0.0),
        "PP_traffic_us": traffic.get("PP", 0.0),
        "EP_traffic_us": traffic.get("EP", 0.0),
        "unknown_traffic_us": traffic.get("unknown", 0.0),
        "total_comm_calls": traffic.get("total_calls", 0),
    }

    # Print summary
    total_traffic = sum(v for k, v in result.items() if k.endswith("_us"))
    print("Traffic Distribution Analysis:")
    print(
        f"  DP: {result['DP_traffic_us']:.2f} us ({100 * result['DP_traffic_us'] / max(total_traffic, 1):.1f}%)"
    )
    print(
        f"  TP: {result['TP_traffic_us']:.2f} us ({100 * result['TP_traffic_us'] / max(total_traffic, 1):.1f}%)"
    )
    print(
        f"  PP: {result['PP_traffic_us']:.2f} us ({100 * result['PP_traffic_us'] / max(total_traffic, 1):.1f}%)"
    )
    print(
        f"  EP: {result['EP_traffic_us']:.2f} us ({100 * result['EP_traffic_us'] / max(total_traffic, 1):.1f}%)"
    )
    print(f"  Unknown: {result['unknown_traffic_us']:.2f} us")
    print(f"  Total comm calls: {result['total_comm_calls']}")

    return result


def _analyze_traffic_distribution(directory: str) -> dict[str, float]:
    """Analyze traffic distribution from trace files."""
    traffic = defaultdict(float)
    traffic["total_calls"] = 0

    for path in Path(directory).iterdir():
        if path.name.startswith("kineto_trace") and path.name.endswith(".json"):
            trace_traffic = _analyze_single_trace(path)
            for key, value in trace_traffic.items():
                traffic[key] += value

    return dict(traffic)


def _analyze_single_trace(trace_path: Path) -> dict[str, float]:
    """Analyze a single trace file for traffic distribution."""
    traffic = defaultdict(float)

    try:
        with trace_path.open() as f:
            data = json.load(f)

        events = data.get("traceEvents", [])

        # Build NVTX range context for classification
        nvtx_ranges = _extract_nvtx_ranges(events)

        # Classify each communication kernel
        for event in events:
            cat = event.get("cat", "")
            name = event.get("name", "")
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)

            if not _is_comm_kernel(name, cat):
                continue

            traffic["total_calls"] += 1

            # Classify by NVTX context or kernel name patterns
            category = _classify_comm_op(name, ts, nvtx_ranges)
            traffic[category] += dur

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {trace_path}: {e}")

    return dict(traffic)


def _extract_nvtx_ranges(events: list[dict]) -> list[tuple[float, float, str]]:
    """Extract NVTX ranges that might indicate parallelism type."""
    ranges = []

    for event in events:
        # NVTX events have category "nvtx" or similar
        cat = event.get("cat", "").lower()
        if "nvtx" not in cat and cat != "user_annotation":
            continue

        name = event.get("name", "").lower()
        ts = event.get("ts", 0)
        dur = event.get("dur", 0)

        if ts > 0:
            ranges.append((ts, ts + dur, name))

    return ranges


def _is_comm_kernel(name: str, cat: str) -> bool:
    """Check if an event is a communication kernel."""
    if cat != "kernel":
        return False

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


def _classify_comm_op(name: str, ts: float, nvtx_ranges: list[tuple[float, float, str]]) -> str:
    """Classify a communication operation by parallelism type.

    Uses:
    1. NVTX range context if available
    2. Kernel name patterns
    """
    name_lower = name.lower()

    # Check NVTX context first
    for range_start, range_end, range_name in nvtx_ranges:
        if range_start <= ts <= range_end:
            if "dp" in range_name or "data_parallel" in range_name or "gradient" in range_name:
                return "DP"
            if "tp" in range_name or "tensor_parallel" in range_name:
                return "TP"
            if "pp" in range_name or "pipeline" in range_name:
                return "PP"
            if "ep" in range_name or "expert" in range_name or "moe" in range_name:
                return "EP"

    # Classify by kernel name patterns
    # PP: Point-to-point operations
    if "sendrecv" in name_lower or "send" in name_lower or "recv" in name_lower:
        return "PP"

    # EP: All-to-all (typically used in MoE expert parallelism)
    if "alltoall" in name_lower:
        return "EP"

    # TP vs DP: Both use AllReduce/ReduceScatter/AllGather
    # Heuristic: TP operations tend to be larger (full activation tensors)
    # DP operations tend to be at the end of backward pass
    # Without more context, classify as unknown or make a best guess

    # Default heuristic: AllReduce after backward is likely DP
    # This is imperfect without proper NVTX tagging
    if "allreduce" in name_lower:
        # Could be either DP or TP
        return "unknown"

    if "reducescatter" in name_lower or "allgather" in name_lower:
        # Often used in FSDP (dp_shard) or TP
        return "unknown"

    return "unknown"
