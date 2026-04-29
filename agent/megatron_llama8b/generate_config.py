"""Seed policy for Megatron-LM Llama-3.1-8B on Perlmutter (2 nodes × 4 GPUs).

Starts with TP=4 (fills one node), DP=2 (one rank per node), PP=1.
This matches the user's original working configuration.
"""


def generate_config(workload: dict, environment: dict) -> dict:
    total_gpus = environment.get("total_gpus", 8)
    gpus_per_node = environment.get("gpus_per_node", 4)

    # Baseline: TP fills one node, DP spans nodes, no PP
    tp = min(gpus_per_node, 4)
    pp = 1
    dp = max(1, total_gpus // (tp * pp))

    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": 1,
        "activation_checkpointing": False,
    }
