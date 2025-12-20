#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

# Communication ops we want to surface.
COMM_KEYS = {
    "all-reduce",
    "all_gather",
    "all-gather",
    "all-to-all",
    "reduce-scatter",
    "collective-permute",
    "collective-permute-start",
    "collective-permute-done",
    "send",
    "recv",
}


def is_comm_event(event: dict) -> bool:
    """Return True if the event looks like a communication kernel."""
    name = str(event.get("name", "")).lower()
    category = str(event.get("args", {}).get("hlo_category", "")).lower()
    return any(key in name for key in COMM_KEYS) or category in COMM_KEYS


def comm_type(event: dict) -> str:
    """Return the communication type label for aggregation."""
    name = str(event.get("name", "")).lower()
    category = str(event.get("args", {}).get("hlo_category", "")).lower()
    if category in COMM_KEYS:
        return category
    for key in COMM_KEYS:
        if key in name:
            return key
    return "unknown"


def duration_us(event: dict) -> float:
    """Get duration in microseconds (uses device_duration_ps when available)."""
    args = event.get("args", {})
    if "device_duration_ps" in args:
        return float(args["device_duration_ps"]) / 1e6
    if "dur" in event:
        return float(event["dur"])
    raise KeyError("No duration field found")


def iter_trace_events(path: Path):
    """Yield trace events from a .trace.json.gz file."""
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    # Some traces wrap events under traceEvents, others may be the array itself.
    events = data.get("traceEvents", data)
    for evt in events:
        yield evt


def parse_batch_tp(path: Path) -> tuple[str | None, str | None]:
    """Extract batch size and TP from the model directory name."""
    batch = None
    tp = None
    for part in path.parts:
        if part.startswith("MODEL_"):
            for token in part.split(","):
                if token.startswith("BATCH_"):
                    batch = token.split("_", 1)[1]
                if token.startswith("TP_"):
                    tp = token.split("_", 1)[1]
    return batch, tp


def comm_kernel_breakdown_tpu(trace_path, print_output=False):
    trace_path = Path(trace_path)
    
    if trace_path.is_dir():
        # Find all .json.gz files in the directory recursively
        trace_files = list(trace_path.rglob("*trace.json.gz"))
        if len(trace_files) == 0:
            raise SystemExit(f"No .json.gz trace files found in directory: {trace_path}")
        elif len(trace_files) > 1:
            raise SystemExit(f"Multiple .json.gz trace files found in directory: {trace_path}. Please specify the file directly.")
        trace_path = trace_files[0]
    
    if not trace_path.exists():
        raise SystemExit(f"Trace file not found: {trace_path}")

    if print_output:
        print(f"Processing trace: {trace_path}")

    try:
        events = list(iter_trace_events(trace_path))
    except Exception as e:
        raise SystemExit(f"Error reading trace file: {e}")

    comm_events = [e for e in events if e.get("ph") == "X" and is_comm_event(e)]

    if not comm_events:
        if print_output:
            print("No communication kernels found.")
        return {}

    totals_us = defaultdict(float)
    counts = defaultdict(int)
    for evt in comm_events:
        totals_us[comm_type(evt)] += duration_us(evt)
        counts[comm_type(evt)] += 1

    batch, tp = parse_batch_tp(trace_path)
    grand_total = sum(totals_us.values())

    if print_output:
        print("-" * 80)
        print(f"{'Comm Type':<25} | {'Total (us)':<15} | {'Count':<10} | {'Avg (us)':<15}")
        print("-" * 80)

        for ctype, total in sorted(totals_us.items(), key=lambda kv: kv[1], reverse=True):
            cnt = counts[ctype]
            avg = total / cnt if cnt > 0 else 0
            print(f"{ctype:<25} | {total:<15.3f} | {cnt:<10} | {avg:<15.3f}")

        print("-" * 80)
        print(f"TOTAL communication time: {grand_total:.3f} us across {len(comm_events)} events")
        if batch and tp:
            print(f"Metadata: Batch={batch}, TP={tp}")
        return None
    else:
        # Return dictionary
        breakdown = {}
        for ctype, total in totals_us.items():
            cnt = counts[ctype]
            breakdown[ctype] = {
                "total_us": total,
                "count": cnt,
                "avg_us": total / cnt if cnt > 0 else 0
            }
        
        return {
            "breakdown": breakdown,
            "total_communication_time_us": grand_total,
            "total_events": len(comm_events),
            "metadata": {
                "batch": batch,
                "tp": tp
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute communication kernel breakdown for a TPU trace.")
    parser.add_argument("trace_path", type=str, help="Path to the .trace.json.gz file")
    args = parser.parse_args()

    trace_path = args.trace_path
    comm_kernel_breakdown_tpu(trace_path, print_output=True)
