
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Minimize GPU count to maximize gpu_step_score.
    
    History:
    - 8 GPUs (tp=4, dp=1, pp=2), mbs=1 → 1.517
    - 2 GPUs (tp=2, dp=1, pp=1), mbs=1, AC=True → FAILED (OOM - full model on each TP rank)
    - 4 GPUs (tp=2, dp=1, pp=2), mbs=2 → 2.561 (BEST)
    
    Try 2 GPUs with pp=2 (model split across pipeline stages), tp=1, dp=1.
    Each GPU holds half the model (~4B params = ~8GB in bf16).
    With activation checkpointing + micro_batch=4 for pipeline efficiency.
    """
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    return {
        "tp": 1,
        "dp": 1,
        "pp": 2,
        "micro_batch_size": 4,
        "activation_checkpointing": True,
    }
