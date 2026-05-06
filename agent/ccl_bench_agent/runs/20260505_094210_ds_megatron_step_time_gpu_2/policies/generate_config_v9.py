
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
        # History analysis (9 runs):
        # - PP=1 always OOMs → need PP>=3
        # - EP=1 with dp=2 failed (run 6) → ep must divide dp
        # - TP=1 with PP=3 failed (run 5) → need TP>=2
        # - Best: tp=2, dp=2, pp=3, ep=2, mbs=2, ac=False → score 1.694 (12 GPUs)
        #
        # Next experiment: try dp=1 to use only 6 GPUs (tp=2*dp=1*pp=3)
        # With dp=1, no DP communication needed at all.
        # ep=1 (must divide dp=1, so ep=1 is the only valid choice from [1,2,4])
        # GPU efficiency: 16/6 = 2.667 vs 16/12 = 1.333
        # Risk: without EP, all 64 experts on each GPU. Each GPU handles
        # ~16B/3(pp) ≈ 5.3B params. With bf16 that's ~10.6GB for params.
        # Optimizer states (Adam): 5.3B * 12 bytes ≈ 64GB... won't fit.
        # Actually with tp=2: 5.3B/2 = 2.65B per GPU → 2.65B * 12 ≈ 32GB
        # Plus activations. Tight on 40GB. Try with ac=True.
        #
        # Actually, MoE params: most params are in experts. With 64 experts,
        # each expert is small but total is large. Without EP, all experts
        # are replicated. With ep=1 and dp=1, each GPU has all experts
        # for its pipeline stage, but only a fraction are activated per token.
        #
        # This might OOM. Let's try it with ac=True to be safe.
        # If it fails, we'll go back to the best known config.
        
        tp = 2
        dp = 1
        pp = 3
        ep = 1  # Only valid choice when dp=1 and ep must divide dp
        micro_batch_size = 2
        activation_checkpointing = True  # Safety measure for tight memory
        
        # used_gpus = 2*1*3 = 6
        # batch=8, dp=1, mbs=2 → acc_steps = 4
        # pipeline bubble = (3-1)/4 = 50%
        
    else:
        # Dense models - general adaptive logic
        activation_checkpointing = True
        
        tp = nearest_valid(2, tp_choices)
        pp = nearest_valid(1, pp_choices)
        dp = nearest_valid(1, dp_choices)
        ep = nearest_valid(1, ep_choices)
        micro_batch_size = nearest_valid(2, mbs_choices)
        
        # Estimate if model fits
        num_params = workload.get("num_params", None)
        if num_params and num_params > 10e9:
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
        micro_batch_size = 2
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
