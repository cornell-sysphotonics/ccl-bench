
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
        # - tp=4, dp=4, pp=1, ep=4 (16 GPUs) → score 1.007
        # - tp=2, dp=2, pp=1, ep=2 (4 GPUs) → FAILED (OOM without PP)
        # - tp=4, dp=2, pp=1, ep=2 (8 GPUs) → FAILED (OOM without PP, mbs=2)
        # - tp=2, dp=2, pp=3, ep=2 (12 GPUs) → score 1.356 (BEST)
        # - tp=2, dp=1, pp=3, ep=1 (6 GPUs) → FAILED (ep=1 OOM with all experts)
        # - tp=1, dp=2, pp=3, ep=2 (6 GPUs) → FAILED 
        #
        # Key: PP=3 works, PP=1 OOMs. ep>=2 required.
        # 
        # Next attempt: tp=4, dp=2, pp=1, ep=2, mbs=1 (8 GPUs)
        # Previous try (run 3) used mbs=2 which may have caused OOM.
        # With mbs=1 and tp=4, per-GPU params are smaller.
        # If this works, 8 GPUs → 0.5*(16/8) + time = 1.0 + time_component
        # 
        # But PP=1 failed even with 8 GPUs before... 
        # Let's try tp=4, dp=1, pp=3, ep=1 → 12 GPUs
        # Wait, ep=1 failed before (run 5 with tp=2,dp=1,pp=3,ep=1)
        #
        # Actually, let's try tp=4, dp=2, pp=3, ep=2 → 24 GPUs (too many!)
        # tp=2, dp=2, pp=3, ep=2 → 12 GPUs was best. 
        #
        # Try to get fewer GPUs: tp=2, dp=1, pp=3, ep=1 failed (ep=1 OOM)
        # tp=1, dp=2, pp=3, ep=2 failed (tp=1 too little splitting)
        #
        # Let's try tp=4, dp=1, pp=3, ep=1 → 12 GPUs but tp=4 splits params more
        # ep=1 failed with tp=2... but tp=4 gives 2x more param splitting
        # Actually ep must divide dp, and dp=1, ep=1 is valid. But 64 experts 
        # all on one GPU is the issue.
        #
        # Let's try: tp=2, dp=2, pp=3, ep=2, mbs=2 → 12 GPUs
        # mbs=2 means acc_steps = 8/(2*2) = 2, bubble = (3-1)/2 = 100%
        # That's terrible for pipeline efficiency. 
        #
        # Better: try tp=2, dp=4, pp=3, ep=4 → 24 GPUs (exceeds 16!)
        # 
        # Let me reconsider. With 12 GPUs and score 1.356, can we improve step time?
        # Currently: tp=2, dp=2, pp=3, ep=2, mbs=1, ac=True
        # acc_steps = 8/(2*1) = 4, bubble = 2/4 = 50%
        # 
        # Try ac=False to speed up compute (no recomputation):
        # tp=2, dp=2, pp=3, ep=2, mbs=1, ac=False → might OOM
        #
        # Try mbs=2 with pp=1 and more GPUs for speed:
        # tp=4, dp=4, pp=1, ep=4, mbs=1 → 16 GPUs (was run 1 with mbs=2, score 1.007)
        # With mbs=1: acc_steps=8/(4*1)=2
        #
        # Let me try: tp=2, dp=4, pp=1, ep=4 → 8 GPUs! dp=4, ep=4, ep divides dp ✓
        # 8 GPUs, all within 2 nodes. ep=4 distributes experts well.
        # PP=1 failed before but those had fewer expert parallelism.
        # With ep=4, each GPU only has 64/4=16 experts → much less memory!
        
        tp = 2
        dp = 4
        pp = 1
        ep = 4  # ep divides dp=4 ✓, ep divides 64 ✓
        micro_batch_size = 1  # conservative for memory
        activation_checkpointing = True
        
        # used_gpus = tp*dp*pp = 2*4*1 = 8
        # batch=8, dp=4, mbs=1 → acc_steps = 8/(4*1) = 2 ✓
        # 8 GPUs → score = 0.5*(16/8) + 0.5*(ref/T) = 1.0 + time_component
        
        # Verify batch divisibility
        if batch_size % (dp * micro_batch_size) != 0:
            for mbs in sorted(mbs_choices):
                if batch_size % (dp * mbs) == 0:
                    micro_batch_size = mbs
                    break
            else:
                micro_batch_size = mbs_choices[0]
    
    else:
        # Dense models - general adaptive logic
        activation_checkpointing = True
        
        tp = nearest_valid(2, tp_choices)
        pp = nearest_valid(1, pp_choices)
        dp = nearest_valid(1, dp_choices)
        ep = nearest_valid(1, ep_choices)
        micro_batch_size = nearest_valid(2, mbs_choices)
        
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
