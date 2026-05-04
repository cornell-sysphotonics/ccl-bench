
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Use fewer GPUs to maximize gpu_step_score.
    
    Previous results:
    - 16 GPUs (tp=4, dp=4, pp=1): score 0.5258
    - 1 GPU (tp=1, dp=1, pp=1): FAILED (likely OOM)
    
    Try 4 GPUs: tp=2, dp=2, pp=1
    - tp=2 splits model across 2 GPUs (within node, fast NVLink)
    - dp=2 for data parallelism
    - activation_checkpointing=True to save memory
    - micro_batch_size=4 (batch_size=32, dp=2 → 16 per rank, 4 accumulation steps)
    
    All 4 GPUs fit within a single node (gpus_per_node=4), so all communication
    is intra-node at 600 Gbps.
    """
    config = {}
    
    # Parse config space
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    config["tp"] = 2
    config["dp"] = 2
    config["pp"] = 1
    config["micro_batch_size"] = 4
    config["activation_checkpointing"] = True
    
    return config
