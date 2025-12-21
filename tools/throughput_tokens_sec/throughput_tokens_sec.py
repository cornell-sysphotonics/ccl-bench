import json
import os
from typing import Dict


def metric_cal(directory: str) -> float:
    """
    Calculate throughput in tokens per second from timing stats.

    Args:
        directory (str): Path to the directory containing timing_stats JSON files.

    Returns:
        float: Throughput in tokens per second.
    """

    timing_candidates = [
        os.path.join(directory, "timing_stats_0.json"),
        os.path.join(directory, "timing_stats_rank0.json"),
    ]
    timing_stats_path = next((p for p in timing_candidates if os.path.exists(p)), None)
    if timing_stats_path is None:
        raise FileNotFoundError(f"No timing_stats_*.json found under {directory}")

    # Read workload card / config to get batch size and sequence length
    workload_card_path = os.path.join(directory, "workload_card.yaml")
    config_path = os.path.join(directory, "config.yaml")

    # Load timing statistics
    with open(timing_stats_path, 'r') as f:
        timing_stats = json.load(f)

    if 'avg_iteration_time' not in timing_stats:
        raise KeyError(f"'avg_iteration_time' missing in {timing_stats_path}")
    avg_iter_time = timing_stats['avg_iteration_time']
    if avg_iter_time <= 0:
        raise ValueError(f"avg_iteration_time <= 0 in {timing_stats_path}, cannot compute throughput")

    import yaml
    if os.path.exists(workload_card_path):
        with open(workload_card_path, 'r') as f:
            workload_card = yaml.safe_load(f)
        batch_size = workload_card['workload']['data']['batch_size']
        seq_len = workload_card['workload']['data']['seq_len']
    elif os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        if 'data' not in cfg:
            raise KeyError(f"'data' missing in {config_path}")
        batch_size = cfg['data']['batch_size']
        seq_len = cfg['data']['seq_len']
    else:
        raise FileNotFoundError(f"No workload_card.yaml or config.yaml found under {directory}")

    # Calculate tokens per iteration
    tokens_per_iter = batch_size * seq_len

    # Calculate throughput
    throughput = tokens_per_iter / avg_iter_time

    return throughput
