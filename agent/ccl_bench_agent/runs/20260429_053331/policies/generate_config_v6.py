
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
    # - tp=4, pp=2, dp=1, mbs=2, ac=False → 0.4743 (BEST by far!)
    # - tp=4, dp=2, pp=1, mbs=2, ac=False → 5.764
    # - tp=4, dp=2, pp=1, mbs=1, ac=False → 5.889
    # - tp=4, dp=2, pp=1, mbs=4, ac=False → OOM
    # - tp=2, dp=4, pp=1 → FAILED
    #
    # pp=2 eliminates inter-node allreduce → massive win
    # Now try mbs=4 with pp=2 (memory split across stages, may fit)
    # global_batch=8, dp=1, mbs=4 → 2 microbatches → bubble = 50%
    # vs mbs=2 → 4 microbatches → bubble = 25%
    # mbs=4 has worse bubble but better compute efficiency per microbatch
    
    tp = 4
    pp = 2
    dp = total_gpus // (tp * pp)  # = 1
    
    # Validate
    valid_dps = choices.get("dp", [1, 2, 4, 8])
    if dp not in valid_dps:
        dp = min(valid_dps, key=lambda x: abs(x - dp))
    
    # Try mbs=4 - larger micro-batch for better compute efficiency
    # pp=2 splits model across stages, so less memory per GPU than pp=1
    micro_batch_size = 4
    activation_checkpointing = False
    
    # Verify micro_batch divides per-dp batch
    per_dp_batch = batch_size // dp
    if per_dp_batch % micro_batch_size != 0:
        # fallback to 2 (known working)
        micro_batch_size = 2
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": micro_batch_size,
        "activation_checkpointing": activation_checkpointing,
    }
