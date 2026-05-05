"""Seed: TP=4, PP=1, DP=4, EP=4 (16 GPUs).
Known-working config for DeepSeek-V2-Lite on A100-40GB.
"""

def generate_config(workload: dict, environment: dict) -> dict:
    return {
        "tp": 4,
        "dp": 4,
        "pp": 1,
        "ep": 4,
        "micro_batch_size": 1,
        "activation_checkpointing": True,
    }
