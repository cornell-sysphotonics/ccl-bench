import json
import os


def metric_cal(directory: str) -> float:
    """
    Calculate average iteration wall-clock time.

    Args:
        directory (str): Path to the directory containing timing_stats JSON files.

    Returns:
        float: Average iteration wall-clock time in seconds.
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

    if 'avg_iteration_time' not in timing_stats:
        raise KeyError(f"'avg_iteration_time' missing in {timing_stats_path}")
    return timing_stats['avg_iteration_time']
