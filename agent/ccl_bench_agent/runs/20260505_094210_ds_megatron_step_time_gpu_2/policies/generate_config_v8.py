
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
        # History analysis:
        # - PP=1 always OOMs (runs 2,3,7) → need PP>=3
        # - EP=1 OOMs (run 5) → need EP>=2
        # - Best: tp=2, dp=2, pp=3, ep=2, ac=False → score 1.379 (12 GPUs)
        # - Second: tp=2, dp=2, pp=3, ep=2, ac=True → score 1.356 (12 GPUs)
        # - Baseline: tp=4, dp=4, pp=1, ep=4 → score 1.007 (16 GPUs)
        #
        # Next experiment: try mbs=2 with ac=False
        # batch=8, dp=2, mbs=2 → acc_steps = 2
        # Pipeline bubble fraction = (3-1)/2 = 100% (worse than mbs=1's 50%)
        # But each microbatch is larger → better GPU utilization per microbatch
        # Net effect unclear - worth testing
        #
        # Alternative: try tp=4, dp=2, pp=3, ep=2 → 24 GPUs > 16, invalid
        # tp=1, dp=4, pp=3, ep=4 → 12 GPUs, but tp=1 might OOM per-GPU
        #   model_params/pp = 16B/3 ≈ 5.3B, with tp=1 that's all on one GPU
        #   5.3B * 12 bytes ≈ 64GB → won't fit in 40GB
        #
        # So tp>=2 is necessary with pp=3.
        # 
        # Let's try mbs=2 to see if throughput improves despite worse bubble
        
        tp = 2
        dp = 2
        pp = 3
        ep = 2
        micro_batch_size = 2  # Try mbs=2 (prev best was mbs=1)
        activation_checkpointing = False  # Confirmed works without AC
        
        # used_gpus = 2*2*3 = 12
        # batch=8, dp=2, mbs=2 → acc_steps = 2, bubble = (3-1)/2 = 100%
        # vs mbs=1: acc_steps=4, bubble=50%, but 4 smaller microbatches
        
    else:
        # Dense models - general adaptive logic
        activation_checkpointing = True
        
        tp = nearest_valid(2, tp_choices)
        pp = nearest_valid(1, pp_choices)
        dp = nearest_valid(1, dp_choices)
        ep = nearest_valid(1, ep_choices)
        micro_batch_size = nearest_valid(2, mbs_choices)
        
        # Estimate if model fits
        # For larger models, increase parallelism
        num_params = workload.get("num_params", None)
        if num_params and num_params > 10e9:
            # Need more parallelism for large models
            pp = nearest_valid(3, pp_choices)
            dp = nearest_valid(2, dp_choices)
    
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
        activation_checkpointing = False
    
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
