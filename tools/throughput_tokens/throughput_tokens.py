"""Throughput (tokens/sec) metric calculation.

Calculates training throughput measured in tokens processed per second.
Uses workload card metadata and trace timing information.
"""

import json
from pathlib import Path
from typing import Any

import yaml


def metric_cal(directory: str) -> float:
    """Calculate throughput in tokens per second from trace data.

    Args:
        directory (str): Path to the trace directory containing workload_card.yaml
                        and trace files.

    Returns:
        float: Throughput in tokens per second.
    """
    # Try to load workload card for batch_size, seq_len, iterations
    workload_card = _load_workload_card(directory)
    if workload_card is None:
        print(f"Warning: No workload card found in {directory}, using defaults")
        batch_size = 4
        seq_len = 8192
        iterations = 5
    else:
        batch_size = workload_card.get("workload", {}).get("data", {}).get("batch_size", 4)
        seq_len = workload_card.get("workload", {}).get("data", {}).get("seq_len", 8192)
        iterations = workload_card.get("workload", {}).get("model", {}).get("iteration", 5)

    # Calculate total tokens processed
    total_tokens = batch_size * seq_len * iterations

    # Get total wall clock time from traces
    total_time_sec = _get_total_time_from_traces(directory)

    if total_time_sec <= 0:
        print("Warning: Could not determine wall clock time, returning 0")
        return 0.0

    return total_tokens / total_time_sec


def _load_workload_card(directory: str) -> dict[str, Any] | None:
    """Load workload card YAML from directory."""
    # Try different possible names
    possible_names = [
        "workload_card.yaml",
        "workload_card_tp.yaml",
        "workload_card_pp.yaml",
        "workload_card_dp_tp.yaml",
        "workload_card_dp_pp.yaml",
        "workload_card_3d.yaml",
    ]

    for name in possible_names:
        card_path = Path(directory) / name
        if card_path.exists():
            try:
                with card_path.open() as f:
                    result: dict[str, Any] | None = yaml.safe_load(f)
                    return result
            except Exception as e:
                print(f"Error loading {card_path}: {e}")

    return None


def _get_total_time_from_traces(directory: str) -> float:
    """Extract total execution time from trace files.

    Tries multiple sources:
    1. Kineto trace (Chrome trace format)
    2. PyTorch ET trace
    3. NSys exported JSON

    Returns:
        float: Total time in seconds, or 0 if not found.
    """
    # Try Kineto trace first
    kineto_time = _get_time_from_kineto(directory)
    if kineto_time > 0:
        return kineto_time

    # Try PyTorch ET trace
    torch_et_time = _get_time_from_torch_et(directory)
    if torch_et_time > 0:
        return torch_et_time

    return 0.0


def _get_time_from_kineto(directory: str) -> float:
    """Extract time from Kineto Chrome trace format."""
    # Look for kineto trace files
    for path in Path(directory).iterdir():
        if path.name.startswith("kineto_trace") and path.name.endswith(".json"):
            try:
                with path.open() as f:
                    data = json.load(f)

                events = data.get("traceEvents", [])
                if not events:
                    continue

                # Find min start time and max end time
                min_ts = float("inf")
                max_ts = float("-inf")

                for event in events:
                    ts = event.get("ts", 0)
                    dur = event.get("dur", 0)
                    if ts > 0:
                        min_ts = min(min_ts, ts)
                        max_ts = max(max_ts, ts + dur)

                if min_ts < float("inf") and max_ts > float("-inf"):
                    # Timestamps are in microseconds
                    return (max_ts - min_ts) / 1_000_000.0

            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                print(f"Error reading {path}: {e}")
                continue

    return 0.0


def _get_time_from_torch_et(directory: str) -> float:
    """Extract time from PyTorch Execution Trace format."""
    for path in Path(directory).iterdir():
        if path.name.startswith("torch_et") and path.name.endswith(".json"):
            try:
                with path.open() as f:
                    data = json.load(f)

                # PyTorch ET format varies; try to find duration info
                nodes = data.get("nodes", [])
                if nodes:
                    # Find total duration from all nodes
                    total_dur = 0
                    for node in nodes:
                        dur = node.get("dur", 0)
                        if dur > 0:
                            total_dur = max(total_dur, dur)

                    if total_dur > 0:
                        # Duration is typically in microseconds
                        return total_dur / 1_000_000.0

            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                print(f"Error reading {path}: {e}")
                continue

    return 0.0
