
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Use fewer GPUs to maximize gpu_step_score.
    Try TP=2, PP=1, DP=1 = 2 GPUs with activation checkpointing to fit in memory.
    """
    return {
        "tp": 2,
        "dp": 1,
        "pp": 1,
        "micro_batch_size": 1,
        "activation_checkpointing": True,
    }
