
def generate_config(workload: dict, environment: dict) -> dict:
    total_gpus = environment.get("total_gpus", 8)
    gpus_per_node = environment.get("gpus_per_node", 4)
    batch_size = workload.get("batch_size", 8)
    
    # Build valid choices lookup
    config_space = workload.get("config_space", [])
    choices = {}
    for dim in config_space:
        choices[dim["key"]] = dim["choices"]
    
    # History analysis:
    # - tp=4, dp=2, pp=1, mbs=2, ac=False → 5.764 (BEST)
    # - tp=4, dp=2, pp=1, mbs=1, ac=False → 5.889
    # - tp=4, dp=2, pp=1, mbs=4, ac=False → OOM
    # - tp=4, dp=2, pp=1, mbs=4, ac=True → FAILED (exit 1)
    # - tp=2, dp=4, pp=1 → FAILED (cross-node TP)
    #
    # New attempt: tp=4, pp=2, dp=1, mbs=2, ac=False
    # - tp=4 stays within one node (4 gpus/node)
    # - pp=2 spans the two nodes (point-to-point, not allreduce)
    # - dp=1 eliminates costly inter-node allreduce (25 Gbps bottleneck)
    # - global_batch=8, dp=1, mbs=2 → 4 microbatches → bubble = 25%
    # - Trade: 25% pipeline bubble vs eliminating DP allreduce across slow interconnect
    
    tp = 4
    pp = 2
    dp = total_gpus // (tp * pp)  # = 1
    
    # Validate
    valid_dps = choices.get("dp", [1, 2, 4, 8])
    if dp not in valid_dps:
        dp = min(valid_dps, key=lambda x: abs(x - dp))
    
    micro_batch_size = 2
    activation_checkpointing = False
    
    # Verify micro_batch divides per-dp batch
    per_dp_batch = batch_size // dp
    if per_dp_batch % micro_batch_size != 0:
        micro_batch_size = 1  # fallback
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": micro_batch_size,
        "activation_checkpointing": activation_checkpointing,
    }
