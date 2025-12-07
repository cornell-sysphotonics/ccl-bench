"""Throughput (tokens/sec) metric calculation.

Calculates training throughput measured in tokens processed per second.
Uses workload card metadata and trace timing information.

NVTX Dependency: None
Works with both Kineto (torch profiler) and nsys traces.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

import yaml


def metric_cal(directory: str, profile_mode: str = "auto") -> dict[str, Any]:
    """Calculate throughput in tokens per second from trace data.

    Args:
        directory: Path to the trace directory containing workload_card.yaml
                   and trace files.
        profile_mode: Profile mode - 'torch', 'nsys', or 'auto' (detect).

    Returns:
        Dictionary with throughput metrics:
            - throughput_tokens_per_s: Tokens processed per second
            - total_tokens: Total tokens processed in the trace
            - measured_duration_s: Total measured wall-clock time in seconds
            - num_iterations: Number of iterations in the trace
            - batch_size: Batch size used
            - seq_len: Sequence length used
    """
    # Try to load workload card for batch_size, seq_len, iterations
    workload_card = _load_workload_card(directory)
    if workload_card is None:
        print(f"Warning: No workload card found in {directory}, using defaults", file=sys.stderr)
        batch_size = 4
        seq_len = 8192
        iterations = None  # Will be detected from trace
    else:
        workload = workload_card.get("workload", {})
        data_config = workload.get("data", {})
        model_config = workload.get("model", {})

        batch_size = data_config.get("batch_size", 4)
        seq_len = data_config.get("seq_len", 8192)
        iterations = model_config.get("iteration")

    # Get total wall clock time and iteration count from traces
    trace_info = _get_trace_timing_info(directory, profile_mode)
    total_time_sec = trace_info["duration_s"]
    detected_iterations = trace_info["num_iterations"]

    # Use detected iterations if not specified in workload card
    if iterations is None:
        if detected_iterations > 0:
            iterations = detected_iterations
        else:
            print("Warning: Could not detect iteration count, assuming 1", file=sys.stderr)
            iterations = 1

    # Calculate total tokens processed
    total_tokens = batch_size * seq_len * iterations

    if total_time_sec <= 0:
        print("Warning: Could not determine wall clock time, returning 0", file=sys.stderr)
        return {
            "throughput_tokens_per_s": 0.0,
            "total_tokens": total_tokens,
            "measured_duration_s": 0.0,
            "num_iterations": iterations,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }

    throughput = total_tokens / total_time_sec

    return {
        "throughput_tokens_per_s": throughput,
        "total_tokens": total_tokens,
        "measured_duration_s": total_time_sec,
        "num_iterations": iterations,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


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
                print(f"Error loading {card_path}: {e}", file=sys.stderr)

    return None


def _get_trace_timing_info(directory: str, profile_mode: str = "auto") -> dict[str, Any]:
    """Extract timing information from trace files.

    Tries multiple sources based on profile_mode:
    1. Kineto trace (Chrome trace format) - if profile_mode is 'torch' or 'auto'
    2. NSys exported data - if profile_mode is 'nsys' or 'auto'

    Returns:
        Dictionary with:
            - duration_s: Total duration in seconds
            - num_iterations: Number of detected iterations (from ProfilerStep# events)
    """
    if profile_mode in ("torch", "auto"):
        kineto_info = _get_info_from_kineto(directory)
        if kineto_info["duration_s"] > 0:
            return kineto_info

    if profile_mode in ("nsys", "auto"):
        nsys_info = _get_info_from_nsys(directory)
        if nsys_info["duration_s"] > 0:
            return nsys_info

    return {"duration_s": 0.0, "num_iterations": 0}


def _get_info_from_kineto(directory: str) -> dict[str, Any]:
    """Extract timing info from Kineto Chrome trace format.

    Detects iterations from ProfilerStep# events which don't require NVTX.
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

    for path in trace_files:
        if not path.is_file() or path.suffix != ".json":
            continue
        try:
            with path.open() as f:
                data = json.load(f)

            events = data.get("traceEvents", [])
            if not events:
                continue

            # Find min start time and max end time
            min_ts = float("inf")
            max_ts = float("-inf")

            # Count ProfilerStep# events for iteration detection
            profiler_steps: list[dict[str, Any]] = []

            for event in events:
                ts = event.get("ts", 0)
                dur = event.get("dur", 0)
                name = event.get("name", "")

                if ts > 0:
                    min_ts = min(min_ts, ts)
                    max_ts = max(max_ts, ts + dur)

                # Detect ProfilerStep# events (iteration markers)
                if name.startswith("ProfilerStep#"):
                    profiler_steps.append(event)

            if min_ts < float("inf") and max_ts > float("-inf"):
                duration_s = (max_ts - min_ts) / 1_000_000.0
                num_iterations = len(profiler_steps) if profiler_steps else 0
                return {
                    "duration_s": duration_s,
                    "num_iterations": num_iterations,
                }

        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error reading {path}: {e}", file=sys.stderr)
            continue

    return {"duration_s": 0.0, "num_iterations": 0}


def _query_nsys_sqlite(sqlite_path: Path) -> dict[str, Any] | None:
    """Query a single NSys SQLite file for kernel timing.

    Args:
        sqlite_path: Path to the NSys SQLite file.

    Returns:
        Dictionary with timing info or None if query failed.
    """
    try:
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()

        # Query for GPU kernel timing
        cursor.execute("""
            SELECT MIN(start), MAX(end)
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        """)
        row = cursor.fetchone()
        conn.close()
    except Exception as e:
        print(f"Error reading nsys sqlite {sqlite_path}: {e}", file=sys.stderr)
        return None

    if row and row[0] is not None and row[1] is not None:
        # NSys timestamps are in nanoseconds
        duration_s = (row[1] - row[0]) / 1e9
        return {"duration_s": duration_s, "num_iterations": 0}
    return None


def _get_info_from_nsys(directory: str) -> dict[str, Any]:
    """Extract timing info from NSys trace data.

    Looks for exported sqlite or JSON files from nsys.
    """
    trace_dir = Path(directory)

    # Look for nsys exported files
    # NSys can export to SQLite which can be queried for kernel timing
    sqlite_files = list(trace_dir.glob("*.sqlite"))

    for sqlite_path in sqlite_files:
        result = _query_nsys_sqlite(sqlite_path)
        if result is not None:
            return result

    return {"duration_s": 0.0, "num_iterations": 0}


def _get_time_from_torch_et(directory: str) -> dict[str, Any]:
    """Extract timing info from PyTorch Execution Trace format."""
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
                        return {
                            "duration_s": total_dur / 1_000_000.0,
                            "num_iterations": 0,
                        }

            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                print(f"Error reading {path}: {e}", file=sys.stderr)
                continue

    return {"duration_s": 0.0, "num_iterations": 0}
