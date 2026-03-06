import json
import os

def metric_cal(directory: str) -> float:
    """
    Calculate Average Time To First Token (TTFT) in ms.
    """
    timing_candidates = [
        os.path.join(directory, "timing_stats_0.json"),
        os.path.join(directory, "timing_stats_rank0.json"),
    ]
    timing_stats_path = next((p for p in timing_candidates if os.path.exists(p)), None)
    if timing_stats_path is None:
        raise FileNotFoundError(f"No timing_stats_*.json found under {directory}")

    with open(timing_stats_path, 'r') as f:
        timing_stats = json.load(f)

    if 'ttft_avg' not in timing_stats:
        raise KeyError(f"'ttft_avg' missing in {timing_stats_path}")

    # Return in ms
    return timing_stats['ttft_avg'] * 1000.0
