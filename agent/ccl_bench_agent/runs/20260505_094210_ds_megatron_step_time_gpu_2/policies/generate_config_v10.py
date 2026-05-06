
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
        # History analysis (10 runs):
        # Best: tp=2, dp=2, pp=3, ep=2, mbs=2, ac=False → 1.694 (12 GPUs)
        # 
        # Key findings:
        # - PP>=3 needed (PP=1 OOMs)
        # - EP must be >=2 when dp>=2 (ep=1 causes assignment errors)
        # - dp=1 with ep=1 also fails
        # - ac=False >> ac=True (1.694 vs 1.379!)
        # - mbs=2 > mbs=1 (better GPU utilization beats pipeline bubble)
        #
        # Next: Try tp=1, dp=2, pp=3, ep=2 = 6 GPUs
        # This would give GPU efficiency 16/6 = 2.67 vs 16/12 = 1.33
        # With tp=1, each GPU handles more params per layer but PP=3 splits layers
        # Memory: ~16B params / (tp=1 * pp=3) = 5.3B per stage
        # With EP=2 and dp=2, experts are split across 2 GPUs
        # So expert params per GPU halved. Dense params stay same.
        # 5.3B * ~6 bytes (bf16 + optimizer approx) might be tight...
        # 
        # Previous run 5 (tp=1, dp=1, pp=3, ep=1) failed, but that was dp=1/ep=1.
        # tp=1, dp=2, pp=3, ep=2 is different - it has proper EP.
        # 
        # Risk: tp=1 means no tensor parallelism to reduce per-GPU memory.
        # Let's try it with ac=True as safety, and mbs=2.
        
        tp = 1
        dp = 2
        pp = 3
        ep = 2
        micro_batch_size = 2
        activation_checkpointing = True  # Safety for tp=1 (more memory per GPU)
        
        # used_gpus = 1*2*3 = 6
        # batch=8, dp=2, mbs=2 → acc_steps = 2
        # pipeline bubble = (3-1)/2 = 100%
        
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
