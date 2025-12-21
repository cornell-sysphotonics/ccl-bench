import os
import sys

# Allow running both as a package and a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.trace_analyzer import TraceAnalyzer


def metric_cal(directory: str) -> float:
    """
    Communication overhead (%) computed from kineto_trace_0.json.
    Focuses on TP-related collectives (all-reduce/all-gather/reduce-scatter).
    """
    csv_candidates = sorted(
        [
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if name.startswith("cuda_gpu_trace") and name.endswith(".csv")
        ]
    )
    json_candidates = sorted(
        [
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if name.startswith("kineto_trace_") and name.endswith(".json")
        ]
    )
    trace_candidates = csv_candidates or json_candidates
    if not trace_candidates:
        raise FileNotFoundError(f"No cuda_gpu_trace*.csv or kineto_trace_*.json found under {directory}")

    overheads = []
    for trace_path in trace_candidates:
        analyzer = TraceAnalyzer(trace_path)
        overheads.append(analyzer.calculate_comm_overhead())

    # Average across ranks if multiple traces present
    return sum(overheads) / len(overheads)
