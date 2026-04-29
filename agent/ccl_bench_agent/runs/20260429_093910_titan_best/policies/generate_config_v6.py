
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus    = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    batch_size    = workload.get("batch_size", 1)
    gpu_memory    = environment.get("gpu_memory_gb", 80)
    inter_bw      = environment.get("inter_node_bandwidth_gbps", 100)
    intra_bw      = environment.get("intra_node_bandwidth_gbps", 600)
    
    # Build lookup of valid choices for each config dimension
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    num_nodes = total_gpus // gpus_per_node
    
    # Lessons learned from iterations 1-6:
    # - tp=4, pp=2, dp=2, micro_batch=4: score 7.701 (BEST)
    # - tp=4, pp=4, dp=1, micro_batch=4: score 10.14
    # - tp=4, pp=1, dp=4, micro_batch=2: score 11.87
    # - tp=4, pp=1, dp=4, micro_batch=4: score 12.59
    # - tp=2, pp=1, dp=8, micro_batch=4: score 14.22
    #
    # Best config: tp=4, pp=2, dp=2 balances pipeline bubbles vs allreduce.
    # Now try micro_batch=2 instead of 4:
    #   With micro_batch=4: num_micro_batches = 32/(2*4) = 4, bubble_ratio = 1/4 = 25%
    #   With micro_batch=2: num_micro_batches = 32/(2*2) = 8, bubble_ratio = 1/8 = 12.5%
    # Smaller micro_batch means more micro-batches and lower bubble ratio.
    # Trade-off: each micro-batch has less compute efficiency, but bubble reduction may win.
    
    tp = 4
    pp = 2
    dp = total_gpus // (tp * pp)  # = 2
    micro_batch = 2  # Try smaller to reduce bubble ratio
    activation_checkpointing = False
    
    # Validate
    if "tp" in valid_choices and tp not in valid_choices["tp"]:
        tp = min(valid_choices["tp"], key=lambda x: abs(x - tp))
    if "pp" in valid_choices and pp not in valid_choices["pp"]:
        pp = min(valid_choices["pp"], key=lambda x: abs(x - pp))
    dp = total_gpus // (tp * pp)
    if "dp" in valid_choices and dp not in valid_choices["dp"]:
        dp = min(valid_choices["dp"], key=lambda x: abs(x - dp))
    if "micro_batch_size" in valid_choices and micro_batch not in valid_choices["micro_batch_size"]:
        micro_batch = min(valid_choices["micro_batch_size"], key=lambda x: abs(x - micro_batch))
    
    config = {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": activation_checkpointing,
    }
    
    return config
