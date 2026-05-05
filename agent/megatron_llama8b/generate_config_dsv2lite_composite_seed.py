"""Seed: TP=4, PP=3, DP=1, EP=1 (12 GPUs).
Best config from step-time optimization. Avoids expensive alltoall
by using PP instead of EP. Composite score rewards fewer GPUs:
Score = 0.5*(16/G) + 0.5*(2.633/T)
"""

def generate_config(workload: dict, environment: dict) -> dict:
    return {
        "tp": 4,
        "dp": 1,
        "pp": 3,
        "ep": 1,
        "micro_batch_size": 4,
        "activation_checkpointing": False,
    }
