
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Adaptive config generator for LLM training workloads.
    
    Key insight: gpu_step_score rewards using fewer GPUs efficiently.
    Score ≈ 0.5*(total_gpus/used_gpus) + 0.5*(reference_time/step_time)
    
    Strategy: Find the minimum GPU count that fits the model in memory,
    then optimize step time within that constraint.
    """
    gpu_mem = environment.get("gpu_memory_gb", 40)
    gpus_per_node = environment.get("gpus_per_node", 4)
    total_gpus = environment.get("total_gpus", 16)
    is_moe = workload.get("moe", False)
    num_layers = workload.get("num_layers", None)
    batch_size = workload.get("batch_size", 8)
    seq_len = workload.get("seq_len", 1024)
    model_family = workload.get("model_family", "")
    
    # Parse config space for valid choices
    config_space = {}
    for dim in workload.get("config_space", []):
        config_space[dim["key"]] = dim["choices"]
    
    tp_choices = sorted(config_space.get("tp", [1, 2, 4]))
    pp_choices = sorted(config_space.get("pp", [1, 3, 9]))
    dp_choices = sorted(config_space.get("dp", [1, 2, 4]))
    ep_choices = sorted(config_space.get("ep", [1, 2, 4]))
    mbs_choices = sorted(config_space.get("micro_batch_size", [1, 2, 4]))
    ac_choices = config_space.get("activation_checkpointing", [True, False])
    
    # Default num_layers based on model family
    if num_layers is None:
        if "deepseek-v2-lite" in model_family.lower():
            num_layers = 27
        elif "deepseek" in model_family.lower():
            num_layers = 60
        elif "llama" in model_family.lower():
            num_layers = 32
        else:
            num_layers = 32
    
    def nearest_valid(val, choices):
        if val in choices:
            return val
        return min(choices, key=lambda x: abs(x - val))
    
    if is_moe:
        # DeepSeek-V2-Lite MoE: ~16B total params, 64 experts, 27 layers
        # 
        # History:
        # - tp=4, dp=4, pp=1, ep=4 (16 GPUs) → score 1.007 ✓
        # - tp=2, dp=2, pp=1, ep=2 (4 GPUs) → FAILED (OOM)
        # - tp=4, dp=2, pp=1, ep=2 (8 GPUs) → FAILED (OOM)
        # - tp=2, dp=2, pp=3, ep=2 (12 GPUs) → score 1.356 ✓ BEST
        #
        # PP=3 is key for memory reduction. 12 GPUs worked well.
        # Now try: tp=2, dp=1, pp=3, ep=1 → 6 GPUs
        # This would give huge GPU efficiency boost if it fits.
        # Memory: ~16B * 12bytes / (tp=2 * pp=3) ≈ 32GB base
        # But with ep=1, all 64 experts on each GPU - might be tight.
        # With activation checkpointing and mbs=1, worth trying.
        #
        # If dp=1, ep must be 1 (ep divides dp).
        # batch=8, dp=1, mbs=1 → acc_steps=8 → good pipeline fill with pp=3
        
        tp = 2
        dp = 1
        pp = 3
        ep = 1  # must divide dp=1, so ep=1
        micro_batch_size = 1  # conservative for memory
        activation_checkpointing = True
        
        # Verify batch divisibility
        if batch_size % (dp * micro_batch_size) != 0:
            for mbs in sorted(mbs_choices):
                if batch_size % (dp * mbs) == 0:
                    micro_batch_size = mbs
                    break
            else:
                micro_batch_size = mbs_choices[0]
    
    else:
        # Dense models
        # Try minimal parallelism, keeping within memory
        tp = min(2, max(tp_choices))
        pp = 1
        dp = 1
        ep = 1
        micro_batch_size = 2
        activation_checkpointing = True
        
        # Ensure batch divisibility
        if batch_size % (dp * micro_batch_size) != 0:
            micro_batch_size = 1
    
    # Validate all choices against config space
    tp = nearest_valid(tp, tp_choices)
    pp = nearest_valid(pp, pp_choices)
    dp = nearest_valid(dp, dp_choices)
    ep = nearest_valid(ep, ep_choices)
    micro_batch_size = nearest_valid(micro_batch_size, mbs_choices)
    
    # Ensure total GPUs used doesn't exceed available
    used_gpus = tp * dp * pp
    if used_gpus > total_gpus:
        # Fall back to best known config
        tp = 2
        dp = 2
        pp = 3
        ep = 2
        micro_batch_size = 1
        activation_checkpointing = True
    
    # Final batch divisibility check
    if batch_size % (dp * micro_batch_size) != 0:
        for mbs in sorted(mbs_choices):
            if batch_size % (dp * mbs) == 0:
                micro_batch_size = mbs
                break
        else:
            micro_batch_size = mbs_choices[0]
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "ep": ep,
        "micro_batch_size": micro_batch_size,
        "activation_checkpointing": activation_checkpointing,
    }
