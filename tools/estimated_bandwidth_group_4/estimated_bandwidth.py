
from .common import find_trace_files, parse_run_dir, extract_metrics_from_trace
import os

def estimated_bandwidth(trace_directory: str) -> float:
    """
    Calculates the Estimated Aggregate Bandwidth (GB/s) for the given trace directory.
    """
    traces = find_trace_files(trace_directory)
    if not traces:
        print(f"No trace files found in {trace_directory}")
        return float("nan")

    run_dir, trace_path = traces[0]
    
    # meta = parse_run_dir(run_dir) # Not strictly needed for bw if we just take raw bytes / wall time
    metrics = extract_metrics_from_trace(trace_path)
    
    return metrics["bandwidth_wall_gbs"]
