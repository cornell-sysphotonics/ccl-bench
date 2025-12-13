"""Variability Metrics calculation.

Measures stability and variability across iterations:
- Throughput dispersion (mean, std, percentiles)
- Iteration time dispersion (mean, std, percentiles)
- Straggler score (coefficient of variation across ranks)

NVTX Dependency: None
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


def metric_cal(directory: str, profile_mode: str = "auto") -> dict[str, Any]:
    """Calculate variability metrics from trace data.

    Args:
        directory: Path to the trace directory containing trace files.
        profile_mode: Profile mode - 'torch', 'nsys', or 'auto' (detect).

    Returns:
        Dictionary with variability metrics:
            - throughput_dispersion: Dict with mean, std, p5, p50, p95 for throughput
            - iter_time_dispersion: Dict with mean, std, p5, p50, p95 for iteration time
            - straggler_score: Coefficient of variation across ranks
            - rank_time_stats: Per-rank statistics
    """
    trace_dir = Path(directory)

    # Extract iteration times per rank
    rank_times = _extract_rank_iter_times(trace_dir)

    # Calculate throughput dispersion (if we have workload info)
    throughput_dispersion = _calculate_throughput_dispersion(trace_dir, rank_times)

    # Calculate iteration time dispersion
    iter_time_dispersion = _calculate_iter_time_dispersion(rank_times)

    # Calculate straggler score
    straggler_score = _calculate_straggler_score(rank_times)

    # Per-rank statistics
    rank_time_stats = _calculate_rank_stats(rank_times)

    result = {
        "throughput_dispersion": throughput_dispersion,
        "iter_time_dispersion": iter_time_dispersion,
        "straggler_score": straggler_score,
        "rank_time_stats": rank_time_stats,
    }

    # Print summary
    if iter_time_dispersion.get("mean", 0) > 0:
        print(
            f"  Variability: iter_time mean={iter_time_dispersion['mean']:.1f}ms, "
            f"std={iter_time_dispersion['std']:.1f}ms",
            file=sys.stderr,
        )
    if straggler_score.get("cv", 0) > 0:
        print(
            f"  Variability: straggler CV={straggler_score['cv']:.3f}, "
            f"mean_lag={straggler_score['mean_lag_ms']:.1f}ms",
            file=sys.stderr,
        )

    return result


def _extract_rank_iter_times(trace_dir: Path) -> dict[int, list[float]]:
    """Extract iteration times per rank from trace files."""
    rank_times: dict[int, list[float]] = {}

    # Find trace files
    trace_patterns = [
        "kineto_trace*.json",
        "rank*_trace.json",
        "*trace*.json",
    ]

    trace_files: list[Path] = []
    for pattern in trace_patterns:
        trace_files.extend(trace_dir.glob(pattern))

    profile_dir = trace_dir / "profile_trace"
    if profile_dir.exists():
        for pattern in trace_patterns:
            trace_files.extend(profile_dir.glob(pattern))

    # Deduplicate
    trace_files = list(set(trace_files))

    for trace_path in trace_files:
        if not trace_path.is_file() or trace_path.suffix != ".json":
            continue

        rank = _extract_rank_from_filename(trace_path)
        iter_times = _extract_iter_times_from_trace(trace_path)

        if iter_times:
            rank_times[rank] = iter_times

    return rank_times


def _extract_rank_from_filename(path: Path) -> int:
    """Extract rank number from trace filename."""
    name = path.stem

    match = re.search(r"rank[_\s]*(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1))

    return 0


def _extract_iter_times_from_trace(trace_path: Path) -> list[float]:
    """Extract iteration times from a trace file."""
    iter_times: list[float] = []

    try:
        with trace_path.open() as f:
            data = json.load(f)

        events = data.get("traceEvents", [])

        # Find ProfilerStep# events
        profiler_steps: list[dict[str, Any]] = []
        for event in events:
            name = event.get("name", "")
            ph = event.get("ph", "")

            if name.startswith("ProfilerStep#") and ph == "X":
                ts = event.get("ts", 0)
                dur = event.get("dur", 0)
                if ts > 0 and dur > 0:
                    profiler_steps.append({"ts": ts, "dur": dur})

        if profiler_steps:
            profiler_steps.sort(key=lambda e: e["ts"])
            iter_times = [e["dur"] / 1000.0 for e in profiler_steps]  # us -> ms

        # Fallback: compute from step boundaries
        if not iter_times:
            step_starts: list[tuple[int, float]] = []
            for event in events:
                name = event.get("name", "")
                if name.startswith("ProfilerStep#"):
                    ts = event.get("ts", 0)
                    if ts > 0:
                        try:
                            step_num = int(name.replace("ProfilerStep#", ""))
                            step_starts.append((step_num, ts))
                        except ValueError:
                            pass

            if len(step_starts) > 1:
                step_starts.sort(key=lambda x: x[0])
                for i in range(len(step_starts) - 1):
                    time_diff = step_starts[i + 1][1] - step_starts[i][1]
                    iter_times.append(time_diff / 1000.0)  # us -> ms

    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Error reading {trace_path}: {e}", file=sys.stderr)

    return iter_times


def _calculate_throughput_dispersion(
    trace_dir: Path,
    rank_times: dict[int, list[float]],
) -> dict[str, Any]:
    """Calculate throughput dispersion statistics."""
    # Try to get batch_size and seq_len from workload card
    try:
        possible_names = [
            "workload_card.yaml",
            "workload_card_tp.yaml",
            "workload_card_pp.yaml",
            "workload_card_dp_tp.yaml",
            "workload_card_dp_pp.yaml",
            "workload_card_3d.yaml",
        ]

        workload_card = None
        for name in possible_names:
            card_path = trace_dir / name
            if card_path.exists():
                with card_path.open() as f:
                    workload_card = yaml.safe_load(f)
                    break

        if workload_card:
            workload = workload_card.get("workload", {})
            data_config = workload.get("data", {})
            batch_size = data_config.get("batch_size", 1)
            seq_len = data_config.get("seq_len", 4096)

            # Calculate throughput for each iteration
            throughputs: list[float] = []

            # Use rank 0 times if available, otherwise aggregate
            if 0 in rank_times:
                times = rank_times[0]
            elif rank_times:
                # Use first available rank
                times = list(rank_times.values())[0]
            else:
                return {"mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}

            tokens_per_iter = batch_size * seq_len
            for iter_time_ms in times:
                if iter_time_ms > 0:
                    iter_time_s = iter_time_ms / 1000.0
                    throughput = tokens_per_iter / iter_time_s
                    throughputs.append(throughput)

            if throughputs:
                return _calculate_stats(throughputs)
    except Exception as e:
        print(f"Warning: Could not calculate throughput dispersion: {e}", file=sys.stderr)

    return {"mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}


def _calculate_iter_time_dispersion(
    rank_times: dict[int, list[float]],
) -> dict[str, Any]:
    """Calculate iteration time dispersion statistics."""
    all_times: list[float] = []

    # Aggregate times from all ranks
    for times in rank_times.values():
        all_times.extend(times)

    if not all_times:
        return {"mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}

    return _calculate_stats(all_times)


def _calculate_straggler_score(rank_times: dict[int, list[float]]) -> dict[str, Any]:
    """Calculate straggler score (coefficient of variation across ranks).

    For each iteration, compute max(rank_time) - min(rank_time) across ranks.
    Then compute statistics over all iterations.
    """
    if not rank_times:
        return {
            "mean_lag_ms": 0.0,
            "max_lag_ms": 0.0,
            "cv": 0.0,  # Coefficient of variation
        }

    # Find max number of iterations across ranks
    max_iters = max(len(times) for times in rank_times.values())

    if max_iters == 0:
        return {
            "mean_lag_ms": 0.0,
            "max_lag_ms": 0.0,
            "cv": 0.0,
        }

    # For each iteration, compute lag (max - min across ranks)
    iteration_lags: list[float] = []

    for iter_idx in range(max_iters):
        iter_times: list[float] = []
        for times in rank_times.values():
            if iter_idx < len(times):
                iter_times.append(times[iter_idx])

        if len(iter_times) > 1:
            lag = max(iter_times) - min(iter_times)
            iteration_lags.append(lag)

    if not iteration_lags:
        return {
            "mean_lag_ms": 0.0,
            "max_lag_ms": 0.0,
            "cv": 0.0,
        }

    mean_lag = sum(iteration_lags) / len(iteration_lags)
    max_lag = max(iteration_lags)

    # Coefficient of variation
    if len(iteration_lags) > 1:
        variance = sum((lag - mean_lag) ** 2 for lag in iteration_lags) / (len(iteration_lags) - 1)
        std_lag = variance**0.5
        cv = (std_lag / mean_lag) if mean_lag > 0 else 0.0
    else:
        cv = 0.0

    return {
        "mean_lag_ms": mean_lag,
        "max_lag_ms": max_lag,
        "cv": cv,
        "num_iterations": len(iteration_lags),
    }


def _calculate_rank_stats(rank_times: dict[int, list[float]]) -> dict[int, dict[str, Any]]:
    """Calculate statistics per rank."""
    rank_stats: dict[int, dict[str, Any]] = {}

    for rank, times in rank_times.items():
        if times:
            stats = _calculate_stats(times)
            rank_stats[rank] = stats
        else:
            rank_stats[rank] = {"mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}

    return rank_stats


def _calculate_stats(values: list[float]) -> dict[str, float]:
    """Calculate statistical measures: mean, std, percentiles."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}

    sorted_values = sorted(values)
    n = len(sorted_values)

    mean = sum(values) / n

    # Standard deviation
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = variance**0.5
    else:
        std = 0.0

    # Percentiles
    def percentile(p: float) -> float:
        idx = int(p * (n - 1))
        return sorted_values[idx]

    p5 = percentile(0.05)
    p50 = percentile(0.50)
    p95 = percentile(0.95)

    return {
        "mean": mean,
        "std": std,
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "min": sorted_values[0],
        "max": sorted_values[-1],
    }

