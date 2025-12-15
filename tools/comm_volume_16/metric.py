"""Communication Volume metric calculation.

Measures total bytes sent/received in communication operations,
broken down by parallelism type (TP, DP, PP, EP).

NVTX Dependency: Partial (better accuracy with NVTX, but works heuristically)
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any


# NCCL operator-level event names
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
    "c10d::allreduce_",
    "c10d::reduce_scatter_",
    "c10d::allgather_",
    "c10d::broadcast_",
    "c10d::reduce_",
    "c10d::send",
    "c10d::recv_",
    "c10d::alltoall",
)


def metric_cal(directory: str, profile_mode: str = "auto") -> dict[str, Any]:
    """Calculate communication volume by parallelism type.

    Args:
        directory: Path to the trace directory containing trace files.
        profile_mode: Profile mode - 'torch', 'nsys', or 'auto' (detect).

    Returns:
        Dictionary with communication volume metrics:
            - total_bytes: Total communication bytes
            - tp_bytes: Tensor parallel communication bytes
            - dp_bytes: Data parallel communication bytes
            - pp_bytes: Pipeline parallel communication bytes
            - ep_bytes: Expert parallel communication bytes
            - unknown_bytes: Unclassified communication bytes
            - bytes_per_step: Average bytes per training step
            - gb_per_step: Average GB per training step
            - fractions: Dict with fraction of each type
            - mode: Classification mode used ('nvtx' or 'heuristic')
    """
    volume_data = _analyze_comm_volume(directory)

    # Calculate totals
    tp_bytes = volume_data.get("TP", 0.0)
    dp_bytes = volume_data.get("DP", 0.0)
    pp_bytes = volume_data.get("PP", 0.0)
    ep_bytes = volume_data.get("EP", 0.0)
    unknown_bytes = volume_data.get("unknown", 0.0)
    total_bytes = tp_bytes + dp_bytes + pp_bytes + ep_bytes + unknown_bytes

    # Get number of iterations/steps
    num_steps = volume_data.get("num_steps", 1)
    if num_steps == 0:
        num_steps = 1

    bytes_per_step = total_bytes / num_steps
    gb_per_step = bytes_per_step / (1024**3)

    # Calculate fractions
    if total_bytes > 0:
        fractions = {
            "tp": tp_bytes / total_bytes,
            "dp": dp_bytes / total_bytes,
            "pp": pp_bytes / total_bytes,
            "ep": ep_bytes / total_bytes,
            "unknown": unknown_bytes / total_bytes,
        }
    else:
        fractions = {"tp": 0.0, "dp": 0.0, "pp": 0.0, "ep": 0.0, "unknown": 0.0}

    result = {
        "total_bytes": total_bytes,
        "tp_bytes": tp_bytes,
        "dp_bytes": dp_bytes,
        "pp_bytes": pp_bytes,
        "ep_bytes": ep_bytes,
        "unknown_bytes": unknown_bytes,
        "bytes_per_step": bytes_per_step,
        "gb_per_step": gb_per_step,
        "fractions": fractions,
        "num_steps": num_steps,
        "mode": volume_data.get("mode", "heuristic"),
    }

    # Print summary
    mode = result["mode"]
    print(f"Communication Volume Analysis (mode: {mode}):", file=sys.stderr)
    print(
        f"  Total: {total_bytes / (1024**3):.2f} GB ({gb_per_step:.3f} GB/step)",
        file=sys.stderr,
    )
    print(
        f"  TP: {tp_bytes / (1024**3):.2f} GB ({fractions['tp'] * 100:.1f}%)",
        file=sys.stderr,
    )
    print(
        f"  DP: {dp_bytes / (1024**3):.2f} GB ({fractions['dp'] * 100:.1f}%)",
        file=sys.stderr,
    )
    print(
        f"  PP: {pp_bytes / (1024**3):.2f} GB ({fractions['pp'] * 100:.1f}%)",
        file=sys.stderr,
    )
    print(
        f"  EP: {ep_bytes / (1024**3):.2f} GB ({fractions['ep'] * 100:.1f}%)",
        file=sys.stderr,
    )

    if mode == "heuristic":
        print(
            "  Warning: Using heuristic mode. DP/TP classification may be inaccurate.",
            file=sys.stderr,
        )

    return result


def _analyze_comm_volume(directory: str) -> dict[str, Any]:
    """Analyze communication volume from trace files."""
    volume: dict[str, Any] = {}
    volume["mode"] = "heuristic"
    volume["num_steps"] = 0

    trace_dir = Path(directory)

    # Find trace files
    trace_patterns = [
        "kineto_trace*.json",
        "rank*_trace.json",
        "*trace*.json",
    ]

    trace_files: set[Path] = set()
    for pattern in trace_patterns:
        trace_files.update(trace_dir.glob(pattern))

    profile_dir = trace_dir / "profile_trace"
    if profile_dir.exists():
        for pattern in trace_patterns:
            trace_files.update(profile_dir.glob(pattern))

    # Count iterations
    for trace_path in trace_files:
        if trace_path.is_file() and trace_path.suffix == ".json":
            num_steps = _count_iterations(trace_path)
            if num_steps > volume["num_steps"]:
                volume["num_steps"] = num_steps
            break  # Only need to count once

    # Analyze each trace file
    for trace_path in trace_files:
        if trace_path.is_file() and trace_path.suffix == ".json":
            trace_volume = _analyze_single_trace(trace_path)
            for key, value in trace_volume.items():
                if key == "mode":
                    if value == "nvtx":
                        volume["mode"] = "nvtx"
                else:
                    volume[key] = volume.get(key, 0.0) + value

    return volume


def _count_iterations(trace_path: Path) -> int:
    """Count number of iterations in a trace file."""
    try:
        with trace_path.open() as f:
            data = json.load(f)

        events = data.get("traceEvents", [])

        # Count ProfilerStep# events
        count = 0
        for event in events:
            name = event.get("name", "")
            if name.startswith("ProfilerStep#"):
                count += 1

        return count
    except Exception:
        return 0


def _analyze_single_trace(trace_path: Path) -> dict[str, Any]:
    """Analyze a single trace file for communication volume."""
    volume: dict[str, float] = defaultdict(float)
    mode = "heuristic"

    try:
        with trace_path.open() as f:
            data = json.load(f)

        events = data.get("traceEvents", [])

        # Build NVTX range context
        nvtx_ranges = _extract_nvtx_ranges(events)
        if nvtx_ranges:
            mode = "nvtx"

        # Build coalesced event classification
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

        coalesced_classification: dict[tuple[float, float], str] = {}
        for c_start, c_end, _c_dur in coalesced_events:
            ops_inside = [name for name, ts in c10d_events if c_start <= ts <= c_end]

            has_send_recv = any(
                "send" in op.lower() or "recv" in op.lower() for op in ops_inside
            )
            has_allreduce = any("allreduce" in op.lower() for op in ops_inside)
            has_allgather = any("allgather" in op.lower() for op in ops_inside)
            has_reducescatter = any("reduce_scatter" in op.lower() for op in ops_inside)
            has_alltoall = any("alltoall" in op.lower() for op in ops_inside)

            if has_send_recv and not has_allreduce and not has_allgather:
                coalesced_classification[(c_start, c_end)] = "PP"
            elif has_alltoall:
                coalesced_classification[(c_start, c_end)] = "EP"
            elif has_reducescatter:
                coalesced_classification[(c_start, c_end)] = "DP"
            elif has_allreduce:
                coalesced_classification[(c_start, c_end)] = "unknown"
            else:
                coalesced_classification[(c_start, c_end)] = "unknown"

        # Analyze communication events
        for event in events:
            cat = event.get("cat", "")
            name = event.get("name", "")
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)
            args = event.get("args", {})

            if not _is_comm_kernel(name, cat):
                continue

            # Try to extract actual bytes from args
            bytes_transferred = _extract_bytes_from_event(event, args)

            # If we can't get actual bytes, estimate from duration
            # This is a rough approximation - actual bytes would need kernel args
            if bytes_transferred == 0:
                # Rough estimate: assume bandwidth utilization
                # This is very approximate
                bytes_transferred = dur * 1000  # Rough scaling factor

            # Classify by parallelism type
            if name == "nccl:coalesced":
                category = coalesced_classification.get((ts, ts + dur), "unknown")
            else:
                category = _classify_comm_op(name, ts, dur, nvtx_ranges, mode)

            volume[category] += bytes_transferred

        volume["mode"] = mode

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {trace_path}: {e}", file=sys.stderr)

    return volume


def _extract_bytes_from_event(event: dict[str, Any], args: dict[str, Any]) -> float:
    """Try to extract actual bytes transferred from event args."""
    # Check various possible field names
    byte_fields = [
        "bytes",
        "size",
        "num_bytes",
        "total_bytes",
        "data_size",
        "tensor_size",
        "input_size",
        "output_size",
    ]

    for field in byte_fields:
        if field in args:
            value = args[field]
            if isinstance(value, (int, float)) and value > 0:
                return float(value)

    # Try nested structures
    if "inputs" in args:
        inputs = args["inputs"]
        if isinstance(inputs, list) and len(inputs) > 0:
            # Try to get size from first input
            first_input = inputs[0]
            if isinstance(first_input, dict):
                for field in byte_fields:
                    if field in first_input:
                        value = first_input[field]
                        if isinstance(value, (int, float)) and value > 0:
                            return float(value)

    return 0.0


def _extract_nvtx_ranges(events: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    """Extract NVTX ranges that might indicate parallelism type."""
    ranges: list[tuple[float, float, str]] = []

    parallelism_keywords = [
        "dp",
        "tp",
        "pp",
        "ep",
        "data_parallel",
        "tensor_parallel",
        "pipeline",
        "expert",
        "fsdp",
        "ddp",
        "moe",
        "gradient_sync",
        "all_reduce_grad",
    ]

    for event in events:
        cat = event.get("cat", "").lower()
        if "nvtx" not in cat and cat != "user_annotation":
            continue

        name = event.get("name", "").lower()
        ts = event.get("ts", 0)
        dur = event.get("dur", 0)

        if ts > 0 and any(kw in name for kw in parallelism_keywords):
            ranges.append((ts, ts + dur, name))

    return ranges


def _is_comm_kernel(name: str, cat: str) -> bool:
    """Check if an event is a communication kernel or operation."""
    if cat not in ("kernel", "cpu_op", "user_annotation"):
        return False

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
    """Classify a communication operation by parallelism type."""
    name_lower = name.lower()

    # Use NVTX context if available
    if mode == "nvtx" and nvtx_ranges:
        for range_start, range_end, range_name in nvtx_ranges:
            if range_start <= ts <= range_end:
                range_name_parts = range_name.replace("_", " ").replace(":", " ").split()

                if any(p in ["dp", "data_parallel", "gradient"] for p in range_name_parts):
                    return "DP"
                if any(p in ["tp", "tensor_parallel"] for p in range_name_parts):
                    return "TP"
                if any(p in ["pp", "pipeline"] for p in range_name_parts):
                    return "PP"
                if any(p in ["ep", "expert", "moe"] for p in range_name_parts):
                    return "EP"

    # Heuristic classification
    if "sendrecv" in name_lower or "c10d::send" in name_lower or "c10d::recv" in name_lower:
        return "PP"
    if "alltoall" in name_lower:
        return "EP"
    if "reducescatter" in name_lower:
        return "DP"
    if "allgather" in name_lower or "allreduce" in name_lower:
        return "unknown"  # Ambiguous without NVTX

    return "unknown"
