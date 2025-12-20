"""Calculate training phase breakdown metrics from torch profile traces."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any, cast


_LOGGER = logging.getLogger(__name__)

# Patterns for identifying training phases
FORWARD_PATTERNS = [
    re.compile(r"forward", re.IGNORECASE),
    re.compile(r"^aten::", re.IGNORECASE),  # Forward ops typically don't have "backward"
]

BACKWARD_PATTERNS = [
    re.compile(r"backward", re.IGNORECASE),
    re.compile(r"autograd::engine", re.IGNORECASE),
    re.compile(r"Backward\d*$"),
]

OPTIMIZER_PATTERNS = [
    re.compile(r"optimizer", re.IGNORECASE),
    re.compile(r"adam", re.IGNORECASE),
    re.compile(r"sgd", re.IGNORECASE),
    re.compile(r"step", re.IGNORECASE),
    re.compile(r"zero_grad", re.IGNORECASE),
]

FSDP_PATTERNS = [
    re.compile(r"FSDP::", re.IGNORECASE),
    re.compile(r"fsdp::", re.IGNORECASE),
]


def _load_trace_json(path: Path) -> dict[str, Any] | None:
    """Load a torch trace JSON file."""
    try:
        with path.open() as file:
            return cast("dict[str, Any]", json.load(file))
    except Exception as e:
        _LOGGER.warning("Failed to load trace file %s: %s", path, e)
        return None


def _classify_event(name: str, cat: str) -> str:
    """Classify an event into a training phase."""
    # Check for FSDP operations first
    for pattern in FSDP_PATTERNS:
        if pattern.search(name):
            # Further classify FSDP operations
            if "backward" in name.lower() or "post_backward" in name.lower():
                return "fsdp_backward"
            if "forward" in name.lower() or "pre_forward" in name.lower():
                return "fsdp_forward"
            return "fsdp_other"

    # Check for optimizer
    for pattern in OPTIMIZER_PATTERNS:
        if pattern.search(name):
            return "optimizer"

    # Check for backward pass
    for pattern in BACKWARD_PATTERNS:
        if pattern.search(name):
            return "backward"

    # Check for forward pass (default for aten ops without backward)
    if cat == "cpu_op" and name.startswith("aten::"):
        # Check if it's not a backward operation
        is_backward = any(p.search(name) for p in BACKWARD_PATTERNS)
        if not is_backward:
            return "forward"

    # Check explicit forward patterns
    for pattern in FORWARD_PATTERNS:
        if pattern.search(name) and "backward" not in name.lower():
            return "forward"

    return "other"


def _analyze_rank_phases(trace_data: dict[str, Any]) -> dict[str, Any]:
    """Analyze training phases for a single rank."""
    if "traceEvents" not in trace_data:
        return {}

    events = trace_data["traceEvents"]

    # Phase durations
    phase_durations = {
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
        "fsdp_forward": 0.0,
        "fsdp_backward": 0.0,
        "fsdp_other": 0.0,
        "other": 0.0,
    }

    phase_counts = dict.fromkeys(phase_durations, 0)

    # Track total time
    total_time = 0.0

    for event in events:
        if not isinstance(event, dict):
            continue

        # Only look at duration events
        if event.get("ph") != "X":
            continue

        cat = event.get("cat", "")
        if cat not in ["cpu_op", "user_annotation"]:
            continue

        name = event.get("name", "")
        dur = event.get("dur", 0)

        if dur <= 0:
            continue

        phase = _classify_event(name, cat)
        phase_durations[phase] += dur
        phase_counts[phase] += 1
        total_time += dur

    # Convert to milliseconds and calculate percentages
    result = {
        "total_cpu_time_ms": total_time / 1000,
    }

    for phase, duration in phase_durations.items():
        result[f"{phase}_time_ms"] = duration / 1000
        result[f"{phase}_percentage"] = (
            round(duration / total_time * 100, 2) if total_time > 0 else 0
        )
        result[f"{phase}_count"] = phase_counts[phase]

    # Aggregate FSDP time
    fsdp_total = (
        phase_durations["fsdp_forward"]
        + phase_durations["fsdp_backward"]
        + phase_durations["fsdp_other"]
    )
    result["fsdp_total_time_ms"] = fsdp_total / 1000
    result["fsdp_total_percentage"] = (
        round(fsdp_total / total_time * 100, 2) if total_time > 0 else 0
    )

    # Training phases (excluding FSDP breakdown)
    training_total = (
        phase_durations["forward"] + phase_durations["backward"] + phase_durations["optimizer"]
    )
    result["pure_training_time_ms"] = training_total / 1000

    return result


def _extract_layer_breakdown(trace_data: dict[str, Any]) -> dict[str, float]:
    """Extract time spent per layer from FSDP annotations."""
    if "traceEvents" not in trace_data:
        return {}

    events = trace_data["traceEvents"]
    layer_times: dict[str, float] = {}

    for event in events:
        if not isinstance(event, dict):
            continue

        if event.get("ph") != "X":
            continue

        name = event.get("name", "")
        dur = event.get("dur", 0)

        # Look for layer-specific FSDP operations
        if "layers." in name:
            # Extract layer identifier
            match = re.search(r"layers\.(\d+)", name)
            if match:
                layer_id = f"layer_{match.group(1)}"
                layer_times[layer_id] = layer_times.get(layer_id, 0) + dur

    # Convert to milliseconds
    return {
        k: v / 1000 for k, v in sorted(layer_times.items(), key=lambda x: int(x[0].split("_")[1]))
    }


def metric_cal(
    trace_dir: str,
    workload_card_path: str | None = None,
    profile_mode: str = "auto",
) -> dict[str, Any]:
    """Calculate training phase breakdown metrics from profile traces.

    Args:
        trace_dir: Directory containing profile traces
        workload_card_path: Path to workload card YAML file (optional)
        profile_mode: Profile mode ("torch", "nsys", or "auto")

    Returns:
        Dictionary with phase breakdown metrics:
        {
            "forward_percentage": float,
            "backward_percentage": float,
            "optimizer_percentage": float,
            "fsdp_overhead_percentage": float,
            "per_rank_stats": list,
            "layer_breakdown": dict,
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
    all_layer_breakdowns = []

    for trace_file in sorted(trace_files):
        trace_data = _load_trace_json(trace_file)
        if trace_data is None:
            continue

        rank_stats = _analyze_rank_phases(trace_data)
        rank_stats["rank"] = trace_file.stem
        per_rank_stats.append(rank_stats)

        layer_breakdown = _extract_layer_breakdown(trace_data)
        if layer_breakdown:
            all_layer_breakdowns.append(layer_breakdown)

    if not per_rank_stats:
        return {"error": "Could not analyze any trace files"}

    # Calculate aggregate metrics
    def avg_metric(key: str) -> float:
        values = [r.get(key, 0) for r in per_rank_stats]
        return sum(values) / len(values) if values else 0

    # Aggregate layer breakdown
    avg_layer_breakdown = {}
    if all_layer_breakdowns:
        all_layers: set[str] = set()
        for lb in all_layer_breakdowns:
            all_layers.update(lb.keys())

        for layer in sorted(all_layers, key=lambda x: int(x.split("_")[1])):
            values = [lb.get(layer, 0) for lb in all_layer_breakdowns]
            avg_layer_breakdown[layer] = round(sum(values) / len(values), 2)

    return {
        "avg_forward_time_ms": round(avg_metric("forward_time_ms"), 2),
        "avg_forward_percentage": round(avg_metric("forward_percentage"), 2),
        "avg_backward_time_ms": round(avg_metric("backward_time_ms"), 2),
        "avg_backward_percentage": round(avg_metric("backward_percentage"), 2),
        "avg_optimizer_time_ms": round(avg_metric("optimizer_time_ms"), 2),
        "avg_optimizer_percentage": round(avg_metric("optimizer_percentage"), 2),
        "avg_fsdp_total_time_ms": round(avg_metric("fsdp_total_time_ms"), 2),
        "avg_fsdp_percentage": round(avg_metric("fsdp_total_percentage"), 2),
        "avg_other_time_ms": round(avg_metric("other_time_ms"), 2),
        "avg_other_percentage": round(avg_metric("other_percentage"), 2),
        "layer_breakdown_ms": avg_layer_breakdown,
        "per_rank_stats": per_rank_stats,
        "num_ranks": len(per_rank_stats),
    }
