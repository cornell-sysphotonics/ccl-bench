
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
    
    # Lessons learned from iterations 1-5:
    # - tp=4, pp=1, dp=4, micro_batch=2: score 11.87
    # - tp=2, pp=1, dp=8, micro_batch=4: score 14.22
    # - tp=4, pp=1, dp=4, micro_batch=4: score 12.59
    # - tp=4, pp=2, dp=2, micro_batch=4: score 7.70 (BEST!)
    #
    # Key insight: reducing dp (and thus cross-node allreduce) is critical.
    # Inter-node BW (200 Gbps) is 12x slower than intra-node (2400 Gbps).
    # pp=2 with dp=2 was much better than pp=1 with dp=4.
    #
    # Now try: tp=4, pp=4, dp=1 to completely eliminate allreduce.
    # dp=1 means zero gradient synchronization overhead.
    # pp=4 introduces more pipeline bubbles but with micro_batch_size=4:
    #   global_batch=32, micro_batches_per_step = 32/(1*4) = 8
    #   bubble_ratio = (pp-1)/num_micro_batches = 3/8 = 37.5%
    # vs pp=2: bubble_ratio = 1/4 = 25% (with dp=2, 32/(2*4)=4 micro_batches)
    # The question is: does eliminating allreduce offset the extra bubble?
    
    tp = 4
    pp = 4
    dp = total_gpus // (tp * pp)  # = 1
    micro_batch = 4  # maximize to reduce bubble ratio
    activation_checkpointing = False
    
    # Validate
    if "tp" in valid_choices and tp not in valid_choices["tp"]:
        tp = 4
    if "pp" in valid_choices and pp not in valid_choices["pp"]:
        pp = 2
        dp = total_gpus // (tp * pp)
    if "dp" in valid_choices and dp not in valid_choices["dp"]:
        # fallback
        dp = min(valid_choices["dp"], key=lambda x: abs(x - dp))
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
