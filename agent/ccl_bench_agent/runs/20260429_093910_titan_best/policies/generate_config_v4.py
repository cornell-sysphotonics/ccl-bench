
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
    
    # Lessons learned from iterations 1-4:
    # - tp=4, dp=4, pp=1 is best parallelism so far (score 11.87)
    # - micro_batch=2 is optimal (2 > 4 > 1 for this workload)
    # - tp=2, dp=8 is worse due to more cross-node allreduce
    # 
    # Now exploring: tp=4, pp=2, dp=2
    # Rationale: dp=2 means less cross-node allreduce (only 2 ranks vs 4)
    # pp=2 introduces pipeline bubbles but with micro_batch=4, bubble ratio is smaller
    # tp=4 stays within node (NVLink)
    # If this is worse, we know pp=1 with dp=4 is optimal
    
    # Try tp=4, pp=2, dp=2 with micro_batch=4 to reduce bubble overhead
    tp = 4
    pp = 2
    dp = total_gpus // (tp * pp)  # = 2
    micro_batch = 4  # larger micro_batch reduces pipeline bubble ratio
    activation_checkpointing = False
    
    # Validate
    if "tp" in valid_choices and tp not in valid_choices["tp"]:
        tp = 4
    if "pp" in valid_choices and pp not in valid_choices["pp"]:
        pp = 1
    if "dp" in valid_choices and dp not in valid_choices["dp"]:
        dp = total_gpus // (tp * pp)
    if "micro_batch_size" in valid_choices and micro_batch not in valid_choices["micro_batch_size"]:
        micro_batch = 2
    
    config = {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": activation_checkpointing,
    }
    
    return config
