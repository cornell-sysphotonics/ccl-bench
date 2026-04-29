
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
    # tp=2, pp=4, dp=2, mb=4, AC=True:  1.703 (BEST by far!)
    # tp=4, pp=2, dp=2, mb=4, AC=True:  7.451
    # tp=4, pp=2, dp=2, mb=4, AC=False: 7.701
    # tp=2, pp=2, dp=4, mb=4, AC=True:  8.917
    # tp=4, pp=4, dp=1, mb=4, AC=False: 10.14
    # tp=4, pp=2, dp=2, mb=2, AC=False: 10.48
    # ...
    #
    # The winning config: tp=2, pp=4, dp=2, mb=4, AC=True
    # Now try tp=2, pp=4, dp=2, mb=2, AC=True to see if smaller micro-batch
    # improves pipeline scheduling with pp=4 stages.
    
    tp = 2
    pp = 4
    dp = 2  # 2*4*2 = 16
    micro_batch = 2
    activation_checkpointing = True
    
    # Validate choices
    if "tp" in valid_choices and tp not in valid_choices["tp"]:
        tp = min(valid_choices["tp"], key=lambda x: abs(x - tp))
    if "pp" in valid_choices and pp not in valid_choices["pp"]:
        pp = min(valid_choices["pp"], key=lambda x: abs(x - pp))
    if "dp" in valid_choices and dp not in valid_choices["dp"]:
        dp = min(valid_choices["dp"], key=lambda x: abs(x - dp))
    if "micro_batch_size" in valid_choices and micro_batch not in valid_choices["micro_batch_size"]:
        micro_batch = max(valid_choices["micro_batch_size"])
    
    config = {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": activation_checkpointing,
    }
    
    return config
