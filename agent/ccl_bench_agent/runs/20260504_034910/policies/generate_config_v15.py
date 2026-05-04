
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Best config so far: tp=4, dp=1, pp=1, mbs=2, checkpointing=False → 2.76
    
    Try tp=4, dp=1, pp=1, mbs=4, checkpointing=False
    - Same 4 GPUs (all intra-node, 600 Gbps)
    - Larger micro_batch_size=4 means fewer gradient accumulation steps
      (32/(1*4) = 8 steps vs 32/(1*2) = 16 steps), potentially faster
    - tp=4 splits model across 4 GPUs, so per-GPU memory is lower
    - No checkpointing since we have headroom with tp=4
    """
    config = {}
    
    # Parse config space
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    config["tp"] = 4
    config["dp"] = 1
    config["pp"] = 1
    config["micro_batch_size"] = 4
    config["activation_checkpointing"] = False
    
    return config
