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


def metric_cal(directory: str) -> int:
    """Calculate the total number of NCCL communication calls across all ranks.

    Scans all kineto_trace*.json files in the directory and counts kernel
    events that match known NCCL collective operation patterns.

    Args:
        directory: Path to the directory containing Kineto trace JSON files.

    Returns:
        Total number of communication calls summed across all ranks.
        Returns 0 if no valid trace files are found.
    """
    trace_dir = Path(directory)
    if not trace_dir.exists():
        print(f"Warning: Directory not found: {directory}")
        return 0

    communication_calls = 0
    traces_processed = 0

    # Scan all kineto trace files (one per rank)
    for trace_file in sorted(trace_dir.iterdir()):
        if not (trace_file.name.startswith("kineto_trace") and trace_file.suffix == ".json"):
            continue

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

        except json.JSONDecodeError as e:
            print(f"Warning: Error decoding JSON in {trace_file}: {e}")
            continue

    if traces_processed == 0:
        print(f"Warning: No valid kineto_trace*.json files found in {directory}")

    return communication_calls
