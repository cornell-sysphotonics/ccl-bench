"""Seed: TP=4, DP=4, PP=1, EP=4 (16 GPUs).
16-GPU baseline config. Composite score rewards fewer GPUs:
Score = 0.5*(16/G) + 0.5*(2.633/T)
"""

def generate_config(workload: dict, environment: dict) -> dict:
    return {
        "tp": 4,
        "dp": 4,
        "pp": 1,
        "ep": 4,
        "micro_batch_size": 2,
        "activation_checkpointing": True,
    }
