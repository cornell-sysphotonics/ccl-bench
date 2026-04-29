
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
    # Best: tp=4, pp=2, dp=1, mbs=2, ac=False → 0.4743
    # tp=4, pp=2, dp=1, mbs=1, ac=False → 0.4866 (slightly worse)
    # tp=4, pp=2, dp=1, mbs=4, ac=False → 0.5722 (worse, more bubble)
    #
    # All dp>1 configs with pp=1 either slow (5.7-5.9) or failed
    # pp=2 keeps communication intra-node, avoiding slow inter-node allreduce
    #
    # Try tp=2, pp=2, dp=2, mbs=2:
    # - tp=2 means less TP communication (smaller allreduce)
    # - pp=2 splits model across 2 stages
    # - dp=2 with smart placement could keep DP allreduce intra-node
    # - per_dp_batch = 8/2 = 4, mbs=2 → 2 microbatches
    # - bubble = (2-1)/(2+2-1) = 1/3 ≈ 33% (worse than best's 1/5=20%)
    # 
    # Actually this will likely be worse due to more bubble and inter-node comms.
    # Let me try tp=8, pp=1, dp=1, mbs=2 instead:
    # - All 8 GPUs doing TP (crosses node boundary - 25Gbps)
    # - No pipeline bubble at all
    # - No DP allreduce needed
    # - But TP allreduce crosses nodes (slow)
    #
    # Better idea: try tp=4, pp=2, dp=1, mbs=2 with ac=False (repeat best)
    # but we've already tried that. Let me explore tp=2, pp=4, dp=1, mbs=2.
    # bubble = 3/(4+3) = 43% - too much bubble.
    #
    # Most promising unexplored: tp=2, pp=2, dp=2, mbs=1
    # per_dp_batch = 4, mbs=1 → 4 microbatches
    # bubble = 1/(4+1) = 20% same as best
    # tp=2 less TP comm, but dp=2 adds allreduce
    # If DP allreduce is intra-node, could win
    
    tp = 2
    pp = 2
    dp = total_gpus // (tp * pp)  # = 2
    
    # Validate
    valid_tps = choices.get("tp", [1, 2, 4, 8])
    valid_pps = choices.get("pp", [1, 2, 4])
    valid_dps = choices.get("dp", [1, 2, 4, 8])
    
    if tp not in valid_tps:
        tp = 4
    if pp not in valid_pps:
        pp = 2
    if dp not in valid_dps:
        dp = min(valid_dps, key=lambda x: abs(x - dp))
    
    micro_batch_size = 1
    
    # Verify micro_batch divides per-dp batch
    per_dp_batch = batch_size // dp
    valid_mbs = choices.get("micro_batch_size", [1, 2, 4])
    if per_dp_batch % micro_batch_size != 0:
        # Find largest valid mbs that divides per_dp_batch
        for m in sorted(valid_mbs, reverse=True):
            if per_dp_batch % m == 0:
                micro_batch_size = m
                break
    
    activation_checkpointing = False
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": micro_batch_size,
        "activation_checkpointing": activation_checkpointing,
    }
