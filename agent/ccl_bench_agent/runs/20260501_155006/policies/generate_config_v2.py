
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Use fewer GPUs to maximize gpu_step_score.
    
    Previous results:
    - tp=4, dp=1, pp=2 (8 GPUs) → score 1.517
    - tp=2, dp=1, pp=1 (2 GPUs) → FAILED (likely OOM)
    
    Try tp=2, dp=1, pp=2 = 4 GPUs with micro_batch=2 for better pipeline efficiency.
    4 GPUs should give better gpu_step_score than 8 GPUs if it fits in memory.
    """
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    # Target: 4 GPUs (tp=2, pp=2, dp=1)
    # Each pipeline stage has 2 GPUs with TP=2
    # Model is split across 2 pipeline stages
    # With PP, we want larger micro_batch for better utilization
    return {
        "tp": 2,
        "dp": 1,
        "pp": 2,
        "micro_batch_size": 2,
        "activation_checkpointing": False,
    }
