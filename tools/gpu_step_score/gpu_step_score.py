"""
Metric: gpu_step_score
Description: Composite score balancing GPU efficiency and step time.
    Score = w * (G0 / G) + (1 - w) * (T0 / T)
    where G = total GPUs used, T = avg step time,
    G0 and T0 are baseline references, w is the weight.
Unit: dimensionless (higher is better)
Returns: Float >= 0, or -1 if data unavailable

GPU count is read from gpu_count.txt in the trace directory.
Score parameters (G0, T0, w) are read from gpu_step_score_config.yaml
in the trace directory, with defaults G0=16, T0=0.44, w=0.5.
"""
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avg_step_time.avg_step_time import metric_cal as avg_step_time_cal


def metric_cal(directory: str) -> float:
    """Compute composite score: w * (G0/G) + (1-w) * (T0/T)."""
    # Get avg step time
    T = avg_step_time_cal(directory)
    if T <= 0:
        return -1.0

    # Get GPU count from file written by run script
    gpu_file = os.path.join(directory, "gpu_count.txt")
    if os.path.exists(gpu_file):
        with open(gpu_file) as f:
            G = int(f.read().strip())
    else:
        # Fallback: count rank*_trace.json files
        traces = [f for f in os.listdir(directory)
                  if f.startswith("rank") and f.endswith("_trace.json")]
        G = len(traces) if traces else -1

    if G <= 0:
        return -1.0

    # Load config
    config_file = os.path.join(directory, "gpu_step_score_config.yaml")
    if os.path.exists(config_file):
        with open(config_file) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    G0 = cfg.get("G0", 16)
    T0 = cfg.get("T0", 0.44)
    w = cfg.get("w", 0.5)

    score = w * (G0 / G) + (1 - w) * (T0 / T)
    return score
