
from .common import find_trace_files, parse_run_dir, extract_metrics_from_trace
import os

def mfu(trace_directory: str, peak_tflops_per_chip: float = 918.0) -> float:
    """
    Calculates the Model Flops Utilization (MFU) for the given trace directory.
    MFU = (active_tflops / peak_total) * 100
    peak_total = peak_tflops_per_chip * tp
    """
    traces = find_trace_files(trace_directory)
    if not traces:
        print(f"No trace files found in {trace_directory}")
        return float("nan")

    # Assuming we process the first valid trace we find, or we might need to aggregate?
    # The original script loops over all traces and prints rows. 
    # For a metric tool interacting with 'main.py' which returns a single value, 
    # we should probably return the metric for the *single* trace if only one dir is passed.
    
    # If the user passed a specific trace file or a dir with one trace, we take the first.
    run_dir, trace_path = traces[0] 
    
    meta = parse_run_dir(run_dir)
    metrics = extract_metrics_from_trace(trace_path)
    
    tp = meta["tp"]
    if tp is None:
        # Fallback default or error?
        # print("Warning: TP not found in directory name, assuming TP=1 for calculation")
        tp = 1
        
    peak_total = peak_tflops_per_chip * tp
    
    mfu_active_pct = (metrics["active_tflops"] / peak_total) * 100.0 if peak_total > 0 and metrics["active_tflops"] is not None else 0.0
    
    return mfu_active_pct
