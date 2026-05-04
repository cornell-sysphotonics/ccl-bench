"""Seed: TP=4, PP=1, DP=4 (16 GPUs) with mbs=2.
Known-working config. Composite score rewards fewer GPUs:
Score = 0.5 * (16/G) + 0.5 * (0.44/T)
"""

def generate_config(workload: dict, environment: dict) -> dict:
    return {
        "tp": 4,
        "dp": 4,
        "pp": 1,
        "micro_batch_size": 2,
        "activation_checkpointing": False,
    }
