"""Seed: TP=4, PP=2, DP=1 using 8 GPUs.
Score = 0.5*(16/8) + 0.5*(0.44/T). Fewer GPUs = higher score.
"""

def generate_config(workload: dict, environment: dict) -> dict:
    return {
        "tp": 4,
        "dp": 1,
        "pp": 2,
        "micro_batch_size": 1,
        "activation_checkpointing": False,
    }
