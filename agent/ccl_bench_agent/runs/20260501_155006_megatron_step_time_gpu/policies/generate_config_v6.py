
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Explore 4-GPU configurations to maximize gpu_step_score.
    
    History:
    - 8 GPUs (tp=4, dp=1, pp=2), mbs=1, AC=False → 1.517
    - 2 GPUs (tp=2, dp=1, pp=1), mbs=1, AC=True → FAILED
    - 4 GPUs (tp=2, dp=1, pp=2), mbs=2, AC=False → 2.561
    - 2 GPUs (tp=1, dp=1, pp=2), mbs=4, AC=True → FAILED
    - 4 GPUs (tp=2, dp=1, pp=2), mbs=4, AC=False → FAILED (OOM)
    - 4 GPUs (tp=4, dp=1, pp=1), mbs=2, AC=False → 2.72 (BEST)
    
    Try tp=2, dp=2, pp=1 (4 GPUs), mbs=2, AC=False
    - dp=2 allows processing more data per step
    - tp=2 splits model across 2 GPUs (within node, fast NVLink)
    - No pipeline bubbles (pp=1)
    - mbs=2 is proven safe for memory
    """
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    total_gpus = environment.get("total_gpus", 16)
    gpus_per_node = environment.get("gpus_per_node", 4)
    
    # Try TP=2, DP=2 within a single node - data parallelism + tensor parallelism
    return {
        "tp": 2,
        "dp": 2,
        "pp": 1,
        "micro_batch_size": 2,
        "activation_checkpointing": False,
    }
