import glob
import json
import os
from typing import Dict, List, Tuple


def _iter_trace_files(directory: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(directory, "rank*_trace.json")))
    return files


# Collect individual NCCL kernel durations for each rank.
def collect_kernel_durations(directory: str) -> Tuple[List[str], Dict[str, List[float]]]:
    trace_files = _iter_trace_files(directory)
    rank_durations: Dict[str, List[float]] = {}

    for trace_file in trace_files:
        try:
            with open(trace_file, "r") as f:
                trace_data = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {trace_file}")
            continue
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {trace_file}")
            continue

        for event in trace_data.get("traceEvents", []):
            if event.get("cat", "").lower() != "kernel":
                continue

            name = event.get("name", "")
            lowered_name = name.lower()
            if not lowered_name or not lowered_name.startswith("nccldevkernel_"):
                continue

            duration = event.get("dur")
            pid = event.get("pid")
            if duration is None or pid is None:
                continue

            key = f"{trace_file}:{pid}"
            if key not in rank_durations:
                rank_durations[key] = []
            rank_durations[key].append(duration)

    return trace_files, rank_durations
