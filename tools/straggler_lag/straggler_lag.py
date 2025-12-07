"""Straggler Lag metric calculation.

Measures the relative lag of the slowest device/process in a communication group.
This indicates load imbalance and synchronization overhead.

NVTX Dependency: None
This metric works purely from NCCL kernel events across ranks.

Grouping Strategy:
- Group NCCL kernels across ranks by matching the k-th kernel of each rank
- For each collective instance, compute the lag between earliest and latest start time
- Optionally uses ProfilerStep# events to compute per-iteration straggler metrics
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
import sys
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


def metric_cal(directory: str, profile_mode: str = "auto") -> dict[str, Any]:
    """Calculate straggler lag from multi-rank trace data.

    Args:
        directory: Path to the trace directory containing trace files.
        profile_mode: Profile mode - 'torch', 'nsys', or 'auto' (detect).

    Returns:
        Dictionary with straggler lag metrics:
            - mean_lag_us: Mean lag across all collectives in microseconds
            - p50_lag_us: Median (P50) lag
            - p95_lag_us: 95th percentile lag
            - max_lag_us: Maximum lag observed
            - min_lag_us: Minimum lag observed
            - num_collectives: Number of collective operations analyzed
            - num_ranks: Number of ranks found
            - normalized_lag: Mean lag normalized by avg iteration time (if available)
            - per_rank_end_times_us: End times per rank (for debugging)
    """
    # Find all rank traces
    rank_traces = _find_rank_traces(directory)

    if len(rank_traces) <= 1:
        print("Warning: Single rank or no traces found, straggler lag = 0", file=sys.stderr)
        return {
            "mean_lag_us": 0.0,
            "p50_lag_us": 0.0,
            "p95_lag_us": 0.0,
            "max_lag_us": 0.0,
            "min_lag_us": 0.0,
            "num_collectives": 0,
            "num_ranks": len(rank_traces),
            "normalized_lag": 0.0,
            "per_rank_end_times_us": {},
        }

    # Analyze NCCL kernels across ranks
    result = _analyze_straggler_lag(rank_traces)

    # Print summary
    print("Straggler Analysis:", file=sys.stderr)
    print(f"  Ranks analyzed: {result['num_ranks']}", file=sys.stderr)
    print(f"  Collectives analyzed: {result['num_collectives']}", file=sys.stderr)
    print(f"  Mean lag: {result['mean_lag_us']:.2f} us", file=sys.stderr)
    print(f"  P50 lag: {result['p50_lag_us']:.2f} us", file=sys.stderr)
    print(f"  P95 lag: {result['p95_lag_us']:.2f} us", file=sys.stderr)
    print(f"  Max lag: {result['max_lag_us']:.2f} us", file=sys.stderr)
    if result["normalized_lag"] > 0:
        print(f"  Normalized lag: {result['normalized_lag']:.4f}", file=sys.stderr)

    return result


def _find_rank_traces(directory: str) -> dict[int, str]:
    """Find trace files organized by rank."""
    rank_traces: dict[int, str] = {}
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

    for path in trace_files:
        if not path.is_file() or path.suffix != ".json":
            continue

        # Try to extract rank from filename
        name = path.name.replace(".json", "").lower()

        # Try different patterns: rank0_trace, kineto_trace_0, trace_rank_0
        rank = -1
        if name.startswith("rank"):
            # rank0_trace.json -> rank 0
            try:
                rank_part = name.split("_")[0].replace("rank", "")
                rank = int(rank_part)
            except ValueError:
                pass
        elif "rank" in name:
            # kineto_trace_rank_0.json
            parts = name.split("_")
            for i, part in enumerate(parts):
                if part == "rank" and i + 1 < len(parts):
                    with contextlib.suppress(ValueError):
                        rank = int(parts[i + 1])
                    break
        else:
            # Try last number in filename
            parts = name.replace(".json", "").split("_")
            for part in reversed(parts):
                try:
                    rank = int(part)
                    break
                except ValueError:
                    continue

        if rank >= 0:
            rank_traces[rank] = str(path)

    return rank_traces


def _analyze_straggler_lag(rank_traces: dict[int, str]) -> dict[str, Any]:
    """Analyze straggler lag across all ranks.

    Groups NCCL kernels by their position (k-th kernel on each rank)
    and computes lag for each grouped collective.
    """
    # Extract NCCL kernel times from each rank
    rank_nccl_kernels: dict[int, list[tuple[float, float]]] = {}  # rank -> [(start, end), ...]
    rank_end_times: dict[int, float] = {}
    iteration_times: list[float] = []

    for rank, trace_path in rank_traces.items():
        kernels, end_time, iter_time = _extract_nccl_kernels(trace_path)
        if kernels:
            rank_nccl_kernels[rank] = kernels
            rank_end_times[rank] = end_time
            if iter_time > 0:
                iteration_times.append(iter_time)

    if len(rank_nccl_kernels) <= 1:
        return {
            "mean_lag_us": 0.0,
            "p50_lag_us": 0.0,
            "p95_lag_us": 0.0,
            "max_lag_us": 0.0,
            "min_lag_us": 0.0,
            "num_collectives": 0,
            "num_ranks": len(rank_traces),
            "normalized_lag": 0.0,
            "per_rank_end_times_us": rank_end_times,
        }

    # Find minimum kernel count across ranks (for alignment)
    min_kernel_count = min(len(k) for k in rank_nccl_kernels.values())

    # Compute lag for each collective (k-th kernel across all ranks)
    lags: list[float] = []
    for k in range(min_kernel_count):
        start_times = [rank_nccl_kernels[r][k][0] for r in rank_nccl_kernels]

        # Compute lag as max(start) - min(start) for this collective
        start_lag = max(start_times) - min(start_times)
        lags.append(start_lag)

    if not lags:
        return {
            "mean_lag_us": 0.0,
            "p50_lag_us": 0.0,
            "p95_lag_us": 0.0,
            "max_lag_us": 0.0,
            "min_lag_us": 0.0,
            "num_collectives": 0,
            "num_ranks": len(rank_traces),
            "normalized_lag": 0.0,
            "per_rank_end_times_us": rank_end_times,
        }

    # Compute statistics
    lags_sorted = sorted(lags)
    mean_lag = sum(lags) / len(lags)
    p50_lag = lags_sorted[len(lags_sorted) // 2]
    p95_idx = int(len(lags_sorted) * 0.95)
    p95_lag = lags_sorted[min(p95_idx, len(lags_sorted) - 1)]
    max_lag = max(lags)
    min_lag = min(lags)

    # Normalize by iteration time if available
    normalized_lag = 0.0
    if iteration_times:
        avg_iter_time = sum(iteration_times) / len(iteration_times)
        if avg_iter_time > 0:
            normalized_lag = mean_lag / avg_iter_time

    return {
        "mean_lag_us": mean_lag,
        "p50_lag_us": p50_lag,
        "p95_lag_us": p95_lag,
        "max_lag_us": max_lag,
        "min_lag_us": min_lag,
        "num_collectives": len(lags),
        "num_ranks": len(rank_nccl_kernels),
        "normalized_lag": normalized_lag,
        "per_rank_end_times_us": rank_end_times,
    }


def _extract_nccl_kernels(trace_path: str) -> tuple[list[tuple[float, float]], float, float]:
    """Extract NCCL kernel times from a single rank's trace.

    Returns:
        Tuple of:
            - List of (start_time, end_time) for each NCCL kernel/operation
            - Max end time (for overall trace end)
            - Estimated iteration time (from ProfilerStep# if available)
    """
    nccl_kernels: list[tuple[float, float]] = []
    max_end_time = 0.0
    iter_time = 0.0

    try:
        with Path(trace_path).open() as f:
            data = json.load(f)

        events = data.get("traceEvents", [])
        if not events:
            return [], 0.0, 0.0

        profiler_steps: list[dict[str, Any]] = []

        for event in events:
            name = event.get("name", "")
            cat = event.get("cat", "")
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)

            if ts <= 0:
                continue

            # Track max end time
            max_end_time = max(max_end_time, ts + dur) if dur > 0 else max(max_end_time, ts)

            # Track NCCL kernels (kernel category)
            if (cat == "kernel" and "nccl" in name.lower() and dur > 0) or (
                ((cat == "cpu_op" and dur > 0) or (cat == "user_annotation" and dur > 0))
                and (name.startswith(_NCCL_OPERATOR_NAMES) or name in _NCCL_OPERATOR_NAMES)
            ):
                nccl_kernels.append((ts, ts + dur))

            # Track ProfilerStep# for iteration time
            if name.startswith("ProfilerStep#"):
                profiler_steps.append(event)

        # Sort NCCL kernels by start time
        nccl_kernels.sort(key=lambda x: x[0])

        # Estimate iteration time from ProfilerStep# events
        if profiler_steps:
            profiler_steps.sort(key=lambda e: e.get("ts", 0))
            durations = [e.get("dur", 0) for e in profiler_steps if e.get("dur", 0) > 0]
            if durations:
                iter_time = sum(durations) / len(durations)

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {trace_path}: {e}", file=sys.stderr)
        return [], 0.0, 0.0

    return nccl_kernels, max_end_time, iter_time
