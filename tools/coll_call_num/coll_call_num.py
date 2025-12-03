"""NCCL Communication Call Count metric calculation.

Counts the total number of NCCL collective communication kernel invocations
across all ranks in a distributed training trace.

Supported NCCL operations:
    - AllReduce
    - ReduceScatter
    - AllGather
    - Broadcast
    - Reduce
    - SendRecv (point-to-point)
"""

from __future__ import annotations

import json
from pathlib import Path


# NCCL kernel name prefixes to match
_NCCL_KERNEL_PREFIXES = (
    "ncclDevKernel_AllReduce",
    "ncclDevKernel_ReduceScatter",
    "ncclDevKernel_AllGather",
    "ncclDevKernel_Broadcast",
    "ncclDevKernel_Reduce",
    "ncclDevKernel_SendRecv",
    # Also match newer NCCL naming patterns
    "ncclKernel_AllReduce",
    "ncclKernel_ReduceScatter",
    "ncclKernel_AllGather",
    "ncclKernel_Broadcast",
    "ncclKernel_Reduce",
    "ncclKernel_SendRecv",
)


def _find_trace_files(directory: Path) -> list[Path]:
    """Find all valid trace JSON files in a directory.

    Searches for trace files using multiple patterns to support different
    naming conventions from TorchTitan, PyTorch Profiler, and Kineto:
        - kineto_trace*.json (legacy naming)
        - *trace*.json in profile_trace/ subdirectory (TorchTitan default)
        - trace_rank*.json (alternative naming)

    Args:
        directory: Root directory to search for trace files.

    Returns:
        Sorted list of Path objects for found trace files.
    """
    trace_files: list[Path] = []

    # Pattern 1: Direct kineto_trace*.json files in root
    trace_files.extend(directory.glob("kineto_trace*.json"))

    # Pattern 2: TorchTitan's profile_trace subdirectory
    profile_trace_dir = directory / "profile_trace"
    if profile_trace_dir.exists():
        trace_files.extend(profile_trace_dir.glob("*trace*.json"))
        trace_files.extend(profile_trace_dir.glob("*.json"))

    # Pattern 3: Recursive search for any trace JSON (fallback)
    if not trace_files:
        trace_files.extend(directory.rglob("*trace*.json"))

    # Filter out non-trace files and deduplicate
    valid_traces = []
    seen = set()
    for f in trace_files:
        if f.is_file() and f.suffix == ".json" and f not in seen:
            # Skip obviously wrong files
            if "metadata" in f.name.lower():
                continue
            seen.add(f)
            valid_traces.append(f)

    return sorted(valid_traces)


def metric_cal(directory: str) -> int:
    """Calculate the total number of NCCL communication calls across all ranks.

    Scans all trace JSON files in the directory (including profile_trace/
    subdirectory) and counts kernel events that match known NCCL collective
    operation patterns.

    Args:
        directory: Path to the directory containing trace JSON files.

    Returns:
        Total number of communication calls summed across all ranks.
        Returns 0 if no valid trace files are found.
    """
    trace_dir = Path(directory)
    if not trace_dir.exists():
        print(f"Warning: Directory not found: {directory}")
        return 0

    # Find all trace files using flexible pattern matching
    trace_files = _find_trace_files(trace_dir)

    if not trace_files:
        print(f"Warning: No trace JSON files found in {directory}")
        print("  Searched patterns: kineto_trace*.json, profile_trace/*trace*.json")
        return 0

    communication_calls = 0
    traces_processed = 0

    for trace_file in trace_files:
        try:
            with trace_file.open() as f:
                trace_data = json.load(f)

            # Count NCCL kernel events
            for event in trace_data.get("traceEvents", []):
                if event.get("cat") != "kernel":
                    continue

                name = event.get("name", "")
                if name.startswith(_NCCL_KERNEL_PREFIXES):
                    communication_calls += 1

            traces_processed += 1
            print(f"  Processed: {trace_file.name}")

        except json.JSONDecodeError as e:
            print(f"Warning: Error decoding JSON in {trace_file}: {e}")
            continue

    print(f"Processed {traces_processed} trace files")
    return communication_calls
