"""Communication/Computation Overlap metric calculation.

Measures the overlap ratio between communication (NCCL) and computation (CUDA kernels)
on the GPU timeline.

NVTX Dependency: None
This metric works purely from kernel events and doesn't require NVTX instrumentation.

Classification:
- Communication: Kernels with names containing 'nccl'
- Computation: GPU kernels like matmul, attention, gemm, cutlass, sgemm, mma, etc.
- Ignored: memcpy, memset (trivial kernels)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# NCCL kernel patterns (communication)
_COMM_KERNEL_PATTERNS = (
    "nccl",
    "c10d::",
)

# Compute kernel patterns (includes common GPU compute kernels)
_COMPUTE_KERNEL_PATTERNS = (
    "matmul",
    "gemm",
    "cutlass",
    "sgemm",
    "dgemm",
    "hgemm",
    "mma",
    "attention",
    "softmax",
    "layernorm",
    "linear",
    "conv",
    "relu",
    "gelu",
    "aten::",
    "ampere",
    "volta",
    "sm80",
    "sm90",
)

# Patterns to ignore (trivial operations)
_IGNORE_PATTERNS = (
    "memcpy",
    "memset",
    "cudaLaunch",
    "cudaDeviceSynchronize",
)


def metric_cal(directory: str, profile_mode: str = "auto") -> dict[str, Any]:
    """Calculate communication/computation overlap ratio from trace data.

    Args:
        directory: Path to the trace directory containing trace files.
        profile_mode: Profile mode - 'torch', 'nsys', or 'auto' (detect).

    Returns:
        Dictionary with overlap metrics:
            - comm_time_ms: Total communication time in milliseconds
            - comp_time_ms: Total computation time in milliseconds
            - overlap_time_ms: Time where comm and comp overlap
            - overlap_ratio_of_comm: Fraction of comm time that overlaps with comp
            - overlap_ratio_of_comp: Fraction of comp time that overlaps with comm
            - num_comm_kernels: Number of communication kernels
            - num_comp_kernels: Number of computation kernels
    """
    trace_dir = Path(directory)

    # Look for trace files with various naming patterns
    trace_patterns = [
        "kineto_trace*.json",
        "rank*_trace.json",
        "*trace*.json",
    ]

    trace_files: list[Path] = []
    for pattern in trace_patterns:
        trace_files.extend(trace_dir.glob(pattern))

    # Also check profile_trace subdirectory
    profile_dir = trace_dir / "profile_trace"
    if profile_dir.exists():
        for pattern in trace_patterns:
            trace_files.extend(profile_dir.glob(pattern))

    # Process the first valid trace file found
    for path in trace_files:
        if path.is_file() and path.suffix == ".json":
            result = _calculate_overlap_from_trace(path)
            if result is not None:
                return result

    print("Warning: Could not calculate overlap, no suitable traces found", file=sys.stderr)
    return {
        "comm_time_ms": 0.0,
        "comp_time_ms": 0.0,
        "overlap_time_ms": 0.0,
        "overlap_ratio_of_comm": 0.0,
        "overlap_ratio_of_comp": 0.0,
        "num_comm_kernels": 0,
        "num_comp_kernels": 0,
    }


def _is_comm_kernel(name: str) -> bool:
    """Check if a kernel name indicates a communication kernel."""
    name_lower = name.lower()
    return any(p in name_lower for p in _COMM_KERNEL_PATTERNS)


def _is_compute_kernel(name: str, cat: str) -> bool:
    """Check if a kernel name indicates a computation kernel."""
    name_lower = name.lower()

    # Skip trivial operations
    if any(p in name_lower for p in _IGNORE_PATTERNS):
        return False

    # GPU kernel category
    if cat == "kernel":
        # If it's a kernel but not NCCL, it's likely compute
        if not _is_comm_kernel(name):
            return True

    # Check for known compute patterns
    return any(p in name_lower for p in _COMPUTE_KERNEL_PATTERNS)


def _calculate_overlap_from_trace(trace_path: Path) -> dict[str, Any] | None:
    """Calculate overlap from Kineto Chrome trace.

    Returns overlap metrics dictionary or None if unable to calculate.
    """
    try:
        with trace_path.open() as f:
            data = json.load(f)

        events = data.get("traceEvents", [])
        if not events:
            return None

        # Separate communication and computation events
        comm_intervals: list[tuple[float, float]] = []
        comp_intervals: list[tuple[float, float]] = []

        for event in events:
            name = event.get("name", "")
            cat = event.get("cat", "")
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)

            if ts <= 0 or dur <= 0:
                continue

            # Classify kernel
            if _is_comm_kernel(name):
                comm_intervals.append((ts, ts + dur))
            elif _is_compute_kernel(name, cat):
                comp_intervals.append((ts, ts + dur))

        if not comm_intervals:
            print(f"Warning: No communication kernels found in {trace_path.name}", file=sys.stderr)
            return {
                "comm_time_ms": 0.0,
                "comp_time_ms": sum(end - start for start, end in comp_intervals) / 1000.0,
                "overlap_time_ms": 0.0,
                "overlap_ratio_of_comm": 0.0,
                "overlap_ratio_of_comp": 0.0,
                "num_comm_kernels": 0,
                "num_comp_kernels": len(comp_intervals),
            }

        if not comp_intervals:
            print(f"Warning: No computation kernels found in {trace_path.name}", file=sys.stderr)
            return {
                "comm_time_ms": sum(end - start for start, end in comm_intervals) / 1000.0,
                "comp_time_ms": 0.0,
                "overlap_time_ms": 0.0,
                "overlap_ratio_of_comm": 0.0,
                "overlap_ratio_of_comp": 0.0,
                "num_comm_kernels": len(comm_intervals),
                "num_comp_kernels": 0,
            }

        # Calculate overlap using sweep line algorithm
        overlap_time = _calculate_interval_overlap(comm_intervals, comp_intervals)
        total_comm_time = sum(end - start for start, end in comm_intervals)
        total_comp_time = sum(end - start for start, end in comp_intervals)

        # Convert to milliseconds (timestamps are in microseconds)
        comm_time_ms = total_comm_time / 1000.0
        comp_time_ms = total_comp_time / 1000.0
        overlap_time_ms = overlap_time / 1000.0

        # Calculate ratios (avoid division by zero)
        overlap_ratio_of_comm = overlap_time / total_comm_time if total_comm_time > 0 else 0.0
        overlap_ratio_of_comp = overlap_time / total_comp_time if total_comp_time > 0 else 0.0

        print(f"Overlap analysis for {trace_path.name}:", file=sys.stderr)
        print(f"  Comm kernels: {len(comm_intervals)}, Comp kernels: {len(comp_intervals)}", file=sys.stderr)
        print(f"  Comm time: {comm_time_ms:.2f} ms, Comp time: {comp_time_ms:.2f} ms", file=sys.stderr)
        print(f"  Overlap: {overlap_time_ms:.2f} ms ({overlap_ratio_of_comm * 100:.1f}% of comm)", file=sys.stderr)

        return {
            "comm_time_ms": comm_time_ms,
            "comp_time_ms": comp_time_ms,
            "overlap_time_ms": overlap_time_ms,
            "overlap_ratio_of_comm": min(overlap_ratio_of_comm, 1.0),
            "overlap_ratio_of_comp": min(overlap_ratio_of_comp, 1.0),
            "num_comm_kernels": len(comm_intervals),
            "num_comp_kernels": len(comp_intervals),
        }

    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Error reading {trace_path}: {e}", file=sys.stderr)
        return None


def _calculate_interval_overlap(
    intervals_a: list[tuple[float, float]], intervals_b: list[tuple[float, float]]
) -> float:
    """Calculate total overlap time between two sets of intervals.

    Uses a sweep line algorithm for efficiency.
    """
    if not intervals_a or not intervals_b:
        return 0.0

    # Create events: (time, event_type, set)
    # event_type: 1 = start, -1 = end
    events = []

    for start, end in intervals_a:
        events.append((start, 1, "a"))
        events.append((end, -1, "a"))

    for start, end in intervals_b:
        events.append((start, 1, "b"))
        events.append((end, -1, "b"))

    # Sort by time, with starts before ends at same time
    events.sort(key=lambda x: (x[0], -x[1]))

    overlap_time = 0.0
    active_a = 0
    active_b = 0
    last_time = 0.0

    for time, event_type, set_id in events:
        # If both sets were active, add overlap time
        if active_a > 0 and active_b > 0:
            overlap_time += time - last_time

        # Update active counts
        if set_id == "a":
            active_a += event_type
        else:
            active_b += event_type

        last_time = time

    return overlap_time
