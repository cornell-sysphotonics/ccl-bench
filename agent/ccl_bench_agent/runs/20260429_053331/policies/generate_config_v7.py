
def generate_config(workload: dict, environment: dict) -> dict:
    total_gpus = environment.get("total_gpus", 8)
    gpus_per_node = environment.get("gpus_per_node", 4)
    batch_size = workload.get("batch_size", 8)
    
    # Build valid choices lookup
    config_space = workload.get("config_space", [])
    choices = {}
    for dim in config_space:
        choices[dim["key"]] = dim["choices"]
    
    # History analysis (pp=2 configs dominate):
    # - tp=4, pp=2, dp=1, mbs=2, ac=False → 0.4743 (BEST)
    # - tp=4, pp=2, dp=1, mbs=4, ac=False → 0.5722
    #
    # pp=2 avoids inter-node allreduce → massive speedup
    # mbs=2 > mbs=4 because fewer pipeline bubbles (25% vs 50%)
    # Try mbs=1: 8 microbatches → 12.5% bubble, may be even better
    
    tp = 4
    pp = 2
    dp = total_gpus // (tp * pp)  # = 1
    
    # Validate
    valid_dps = choices.get("dp", [1, 2, 4, 8])
    if dp not in valid_dps:
        dp = min(valid_dps, key=lambda x: abs(x - dp))
    
    # Try mbs=1 - more microbatches means less pipeline bubble
    # global_batch=8, dp=1, mbs=1 → 8 microbatches → bubble_fraction = (pp-1)/(num_microbatches+pp-1) = 1/9 ≈ 11%
    micro_batch_size = 1
    activation_checkpointing = False
    
    # Verify micro_batch divides per-dp batch
    per_dp_batch = batch_size // dp
    if per_dp_batch % micro_batch_size != 0:
        micro_batch_size = 2  # fallback to known best
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": micro_batch_size,
        "activation_checkpointing": activation_checkpointing,
    }
