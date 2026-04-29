
def generate_config(workload: dict, environment: dict) -> dict:
    total_gpus = environment.get("total_gpus", 8)
    gpus_per_node = environment.get("gpus_per_node", 4)
    batch_size = workload.get("batch_size", 8)
    
    # Build valid choices lookup
    config_space = workload.get("config_space", [])
    choices = {}
    for dim in config_space:
        choices[dim["key"]] = dim["choices"]
    
    # Key learnings from history:
    # - tp=4, dp=2, pp=1, mbs=2 → 5.764 (best)
    # - tp=4, dp=2, pp=1, mbs=1 → 5.889
    # - tp=4, dp=2, pp=1, mbs=4 → OOM (SIGSEGV)
    # - tp=2, dp=4, pp=1, mbs=2 → FAILED (cross-node TP)
    #
    # Try: tp=4, pp=2, dp=1, mbs=4
    # With pp=2, each stage has half the layers → less memory per GPU
    # tp=4 stays within one node (4 gpus/node), pp=2 spans across nodes
    # dp=1 means no data parallelism
    # global_batch=8, dp=1, so 8 samples per rank
    # With mbs=4, num_microbatches = 8/4 = 2
    # Pipeline bubble fraction with pp=2 and 2 microbatches = (2-1)/(2) = 50% which is bad
    
    # Actually, let me try tp=4, dp=2, pp=1, mbs=2 with activation_checkpointing=True
    # to see if it helps memory-bound situations. If we can reduce memory pressure,
    # the GPU might overlap better. But likely it adds overhead.
    
    # Better idea: just try the known-best config but explore activation_checkpointing=True
    # to see if it matters. Since mbs=2 already works fine, the overhead likely hurts.
    
    # Let me try tp=4, pp=2, dp=1, mbs=2 instead - pipeline parallelism reduces
    # per-stage memory, might allow better compute utilization
    # num_microbatches = 8 / 1 / 2 = 4 microbatches with mbs=2
    # bubble fraction = (2-1)/4 = 25%
    
    # Actually the safest improvement attempt: try activation_checkpointing with mbs=4
    # to avoid OOM. tp=4, dp=2, pp=1, mbs=4, activation_checkpointing=True
    # Activation checkpointing halves activation memory, might let mbs=4 fit
    
    tp = 4
    pp = 1
    dp = total_gpus // (tp * pp)  # = 2
    
    # Validate dp
    valid_dps = choices.get("dp", [1, 2, 4, 8])
    if dp not in valid_dps:
        dp = min(valid_dps, key=lambda x: abs(x - dp))
    
    micro_batch_size = 4
    activation_checkpointing = True  # Enable to avoid OOM with mbs=4
    
    # Verify micro_batch divides per-dp batch
    per_dp_batch = batch_size // dp
    if per_dp_batch % micro_batch_size != 0:
        micro_batch_size = 2  # fallback
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": micro_batch_size,
        "activation_checkpointing": activation_checkpointing,
    }
