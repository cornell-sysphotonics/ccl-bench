
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Explore 4-GPU configurations to maximize gpu_step_score.
    
    History:
    - 8 GPUs (tp=4, dp=1, pp=2), mbs=1, AC=False → 1.517
    - 2 GPUs (tp=2, dp=1, pp=1), mbs=1, AC=True → FAILED
    - 4 GPUs (tp=2, dp=1, pp=2), mbs=2, AC=False → 2.561 (BEST)
    - 2 GPUs (tp=1, dp=1, pp=2), mbs=4, AC=True → FAILED
    - 4 GPUs (tp=2, dp=1, pp=2), mbs=4, AC=False → FAILED (OOM with larger mbs)
    
    Try tp=4, dp=1, pp=1 (4 GPUs), mbs=2, AC=False
    - Pure tensor parallelism eliminates pipeline bubble overhead
    - tp=4 within a single node (4 GPUs/node) uses fast NVLink
    - mbs=2 known to fit in memory with 4 GPUs
    - No pipeline stages means no bubble waste
    """
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    total_gpus = environment.get("total_gpus", 16)
    gpus_per_node = environment.get("gpus_per_node", 4)
    
    # Try pure TP within a single node - no pipeline bubbles, fast intra-node comm
    return {
        "tp": 4,
        "dp": 1,
        "pp": 1,
        "micro_batch_size": 2,
        "activation_checkpointing": False,
    }
