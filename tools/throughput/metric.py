"""Calculate throughput (tokens per second) from torch profile traces."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)


def _load_trace_json(path: Path) -> dict[str, Any] | None:
    """Load a torch trace JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        _LOGGER.warning(f"Failed to load trace file {path}: {e}")
        return None


def _find_iteration_time(trace_data: dict[str, Any]) -> float | None:
    """
    Extract iteration time from torch trace.
    
    Looks for iteration markers or calculates from first to last event.
    Returns time in seconds.
    """
    if "traceEvents" not in trace_data:
        return None
    
    events = trace_data["traceEvents"]
    if not events:
        return None
    
    # Find iteration start/end markers
    start_time = None
    end_time = None
    
    for event in events:
        if not isinstance(event, dict):
            continue
        
        name = event.get("name", "")
        ts = event.get("ts")
        
        if ts is None:
            continue
        
        # Look for iteration markers
        if "iteration" in name.lower() and "start" in name.lower():
            start_time = ts
        elif "iteration" in name.lower() and "end" in name.lower():
            end_time = ts
        elif "Record Window End" in name:
            end_time = ts
    
    # If no explicit markers, use first and last event timestamps
    if start_time is None or end_time is None:
        timestamps = [e.get("ts") for e in events if isinstance(e, dict) and e.get("ts") is not None]
        if timestamps:
            start_time = min(timestamps)
            end_time = max(timestamps)
    
    if start_time is None or end_time is None:
        return None
    
    # Convert from microseconds to seconds (Chrome trace format uses microseconds)
    duration_us = end_time - start_time
    duration_s = duration_us / 1_000_000.0
    
    return duration_s


def _get_tokens_per_step(workload_card_path: str | None) -> int:
    """
    Extract tokens per step from workload card.
    
    Returns: tokens per step (global_batch_size * seq_len)
    """
    if workload_card_path is None:
        _LOGGER.warning("No workload card provided, using default values")
        return 8192  # Default fallback
    
    try:
        try:
            import yaml
        except ImportError:
            _LOGGER.warning("PyYAML not installed, cannot parse workload card. Install with: pip install pyyaml")
            return 8192
        
        with open(workload_card_path, "r") as f:
            card = yaml.safe_load(f)
        
        # Extract batch size and sequence length
        batch_size = card.get("workload", {}).get("data", {}).get("batch_size", 8)
        seq_len = card.get("workload", {}).get("data", {}).get("seq_len", 1024)
        
        tokens_per_step = batch_size * seq_len
        _LOGGER.info(f"Loaded from workload card: batch_size={batch_size}, seq_len={seq_len}, tokens_per_step={tokens_per_step}")
        return tokens_per_step
    except Exception as e:
        _LOGGER.warning(f"Failed to load workload card {workload_card_path}: {e}, using defaults")
        return 8192


def metric_cal(
    trace_dir: str,
    workload_card_path: str | None = None,
    profile_mode: str = "auto",
) -> dict[str, Any]:
    """
    Calculate throughput (tokens per second) from profile traces.
    
    Args:
        trace_dir: Directory containing profile traces (e.g., profile_traces/iteration_5/)
        workload_card_path: Path to workload card YAML file
        profile_mode: Profile mode ("torch", "nsys", or "auto")
    
    Returns:
        Dictionary with throughput metrics:
        {
            "throughput_tokens_per_sec": float,
            "tokens_per_step": int,
            "iteration_time_sec": float,
            "num_ranks": int,
            "per_rank_throughput": list[float],
        }
    """
    trace_path = Path(trace_dir)
    
    if not trace_path.exists():
        return {"error": f"Trace directory does not exist: {trace_dir}"}
    
    # Find all trace JSON files
    trace_files = list(trace_path.glob("*trace.json"))
    if not trace_files:
        # Try alternative patterns
        trace_files = list(trace_path.glob("rank*_trace.json"))
    
    if not trace_files:
        return {"error": f"No torch trace JSON files found in {trace_dir}"}
    
    _LOGGER.info(f"Found {len(trace_files)} trace files in {trace_dir}")
    
    # Get tokens per step from workload card
    tokens_per_step = _get_tokens_per_step(workload_card_path)
    
    # Calculate iteration time for each rank
    iteration_times = []
    per_rank_throughput = []
    
    for trace_file in sorted(trace_files):
        trace_data = _load_trace_json(trace_file)
        if trace_data is None:
            continue
        
        iter_time = _find_iteration_time(trace_data)
        if iter_time is None or iter_time <= 0:
            _LOGGER.warning(f"Could not extract iteration time from {trace_file}")
            continue
        
        iteration_times.append(iter_time)
        throughput = tokens_per_step / iter_time
        per_rank_throughput.append(throughput)
    
    if not iteration_times:
        return {"error": "Could not extract iteration times from any trace files"}
    
    # Calculate aggregate metrics
    avg_iter_time = sum(iteration_times) / len(iteration_times)
    total_throughput = tokens_per_step / avg_iter_time
    
    return {
        "throughput_tokens_per_sec": total_throughput,
        "tokens_per_step": tokens_per_step,
        "iteration_time_sec": avg_iter_time,
        "num_ranks": len(iteration_times),
        "per_rank_throughput": per_rank_throughput,
        "per_rank_iteration_time": iteration_times,
        "min_iteration_time": min(iteration_times),
        "max_iteration_time": max(iteration_times),
    }

