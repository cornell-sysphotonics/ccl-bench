
def generate_config(workload: dict, environment: dict) -> dict:
    total_gpus = environment.get("total_gpus", 8)
    gpus_per_node = environment.get("gpus_per_node", 4)
    batch_size = workload.get("batch_size", 8)
    
    # Build valid choices lookup
    config_space = workload.get("config_space", [])
    choices = {}
    for dim in config_space:
        choices[dim["key"]] = dim["choices"]
    
    # Key learnings:
    # - TP must stay within a node (tp <= gpus_per_node) to avoid cross-node TP errors
    # - tp=4, dp=2, pp=1 is stable
    # - micro_batch=2 > micro_batch=1 (5.764 vs 5.889)
    # - Try micro_batch=4: with dp=2, per_dp_batch=4, so 1 accumulation step = minimal overhead
    
    # Strategy: keep TP within node, minimize pipeline stages, maximize micro_batch
    tp = min(gpus_per_node, max(choices.get("tp", [1])))  # = 4
    pp = 1
    dp = total_gpus // (tp * pp)
    
    # Validate dp is in choices
    if dp not in choices.get("dp", [1, 2, 4, 8]):
        # Find closest valid dp
        valid_dps = choices.get("dp", [1, 2, 4, 8])
        dp = min(valid_dps, key=lambda x: abs(x - dp))
    
    # Maximize micro_batch_size to reduce gradient accumulation steps
    per_dp_batch = batch_size // dp
    valid_mbs = sorted(choices.get("micro_batch_size", [1, 2, 4]), reverse=True)
    micro_batch_size = 1
    for mbs in valid_mbs:
        if per_dp_batch >= mbs and per_dp_batch % mbs == 0:
            micro_batch_size = mbs
            break
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": micro_batch_size,
        "activation_checkpointing": False,
    }
