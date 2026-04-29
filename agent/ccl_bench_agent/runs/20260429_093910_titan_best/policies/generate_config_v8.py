
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
    # tp=4, pp=2, dp=2, mb=4, AC=False: 7.701 (BEST)
    # tp=4, pp=4, dp=1, mb=4, AC=False: 10.14
    # tp=4, pp=2, dp=2, mb=2, AC=False: 10.48
    # tp=4, pp=1, dp=4, mb=2, AC=False: 11.87
    # tp=4, pp=1, dp=4, mb=4, AC=False: 11.92 / 12.59
    # tp=2, pp=1, dp=8, mb=4, AC=False: 14.22
    # tp=2, pp=2, dp=4, mb=1, AC=False: 15.28
    #
    # Key insights:
    # - tp=4 is clearly better than tp=2 (keeps comm on NVLink within node)
    # - pp=2 with dp=2 is the best parallelism split
    # - mb=4 is better than mb=2 (fewer pipeline bubbles)
    # - AC=True not yet tested with best config
    #
    # Try: tp=4, pp=2, dp=2, mb=4, AC=True
    # This tests if activation checkpointing can improve the best config
    # by reducing memory pressure (though it adds recomputation overhead)
    
    # General strategy for good configs:
    # 1. Maximize TP within a node (use fast NVLink)
    # 2. Use moderate PP (2 stages) to balance pipeline bubbles vs memory
    # 3. Use remaining GPUs for DP
    # 4. Use largest micro_batch to minimize pipeline bubbles
    
    tp = min(gpus_per_node, 4)  # tp=4, fits within a node
    if "tp" in valid_choices and tp not in valid_choices["tp"]:
        tp = max(c for c in valid_choices["tp"] if c <= gpus_per_node)
    
    pp = 2
    if "pp" in valid_choices and pp not in valid_choices["pp"]:
        pp = min(valid_choices["pp"], key=lambda x: abs(x - pp))
    
    dp = total_gpus // (tp * pp)
    if "dp" in valid_choices and dp not in valid_choices["dp"]:
        dp = min(valid_choices["dp"], key=lambda x: abs(x - dp))
    
    # Use largest micro_batch to reduce pipeline bubbles
    micro_batch = 4
    if "micro_batch_size" in valid_choices:
        micro_batch = max(valid_choices["micro_batch_size"])
    
    # Try activation checkpointing with the best config
    activation_checkpointing = True
    
    config = {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": activation_checkpointing,
    }
    
    return config
