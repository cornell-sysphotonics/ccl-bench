
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
    
    # Execution history analysis (minimizing avg_step_time):
    # tp=1, pp=4, dp=4, mb=2, AC=True:  1.500 (BEST!)
    # tp=2, pp=4, dp=2, mb=2, AC=True:  1.578
    # tp=2, pp=4, dp=2, mb=4, AC=True:  1.703
    # tp=2, pp=4, dp=2, mb=1, AC=True:  2.968
    # tp=4, pp=2, dp=2, mb=4, AC=True:  7.451
    # tp=4, pp=2, dp=2, mb=4, AC=False: 7.701
    # tp=2, pp=2, dp=4, mb=4, AC=True:  8.917
    #
    # Key insights:
    # - pp=4 is dominant factor for performance
    # - tp=1 > tp=2 (less communication overhead)
    # - mb=2 is the sweet spot with pp=4
    # - activation_checkpointing=True is needed for memory
    #
    # Try tp=1, pp=4, dp=4, mb=4, AC=True to explore if fewer 
    # gradient accumulation steps could help (only 2 steps vs 4 with mb=2).
    # With pp=4, pipeline needs enough micro-batches to fill; 
    # mb=4 gives 2 acc steps which may not fill pipeline well.
    # But worth exploring since tp=1 frees memory.
    
    tp = 1
    pp = 4
    dp = 4  # 1*4*4 = 16
    micro_batch = 4
    activation_checkpointing = True
    
    # Validate choices
    if "tp" in valid_choices and tp not in valid_choices["tp"]:
        tp = min(valid_choices["tp"], key=lambda x: abs(x - tp))
    if "pp" in valid_choices and pp not in valid_choices["pp"]:
        pp = min(valid_choices["pp"], key=lambda x: abs(x - pp))
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
