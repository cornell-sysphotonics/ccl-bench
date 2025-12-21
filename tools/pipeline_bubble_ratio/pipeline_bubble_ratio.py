import os
import sys

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from tools.trace_analyzer import TraceAnalyzer
except ImportError:
    # Fallback
    from trace_analyzer import TraceAnalyzer

def metric_cal(path: str) -> float:
    """
    Calculate Pipeline Bubble Ratio (%) from trace.
    Supports both Kineto JSON (kineto_trace_0.json) and Nsys CSV (cuda_gpu_trace.csv).
    Args:
        path: Directory containing trace files OR path to a specific trace file.
    """
    trace_paths = []

    if os.path.isdir(path):
        csv_candidates = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.startswith("cuda_gpu_trace") and name.endswith(".csv")
        ]
        json_candidates = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.startswith("kineto_trace_") and name.endswith(".json")
        ]
        trace_paths = sorted(csv_candidates or json_candidates)
    else:
        trace_paths = [path]

    if not trace_paths:
        raise FileNotFoundError(f"No trace files found in or at {path}")

    ratios = []
    for trace_path in trace_paths:
        analyzer = TraceAnalyzer(trace_path)
        ratios.append(analyzer.calculate_bubble_ratio())

    return sum(ratios) / len(ratios)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline_bubble_ratio.py <trace_directory_or_file>")
        sys.exit(1)
        
    path = sys.argv[1]
    ratio = metric_cal(path)
    print(f"Pipeline Bubble Ratio: {ratio:.2f}%")
