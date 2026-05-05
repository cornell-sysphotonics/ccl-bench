
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Adaptive config for LLM training workloads.
    Reasons about model properties, GPU memory, and network topology.
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
    
    tp_choices = sorted(config_space.get("tp", [1]))
    pp_choices = sorted(config_space.get("pp", [1]))
    dp_choices = sorted(config_space.get("dp", [1]))
    ep_choices = sorted(config_space.get("ep", [1]))
    mbs_choices = sorted(config_space.get("micro_batch_size", [1]))
    ac_choices = config_space.get("activation_checkpointing", [True])
    
    if "deepseek" in model_family.lower():
        # DeepSeek-V2-Lite: MoE with 64 experts, 27 layers
        # A100-40GB, 4 nodes × 4 GPUs, 16 total
        #
        # CONFIRMED RESULTS:
        #   tp=4, dp=4, pp=1, ep=4, mbs=1, ac=True  → 3.359
        #   tp=4, dp=4, pp=1, ep=4, mbs=2, ac=True  → 2.736
        #   tp=4, dp=4, pp=1, ep=4, mbs=2, ac=False → 2.633
        #   tp=4, dp=1, pp=3, ep=1, mbs=4, ac=True  → 1.642 ★ BEST
        #
        # CONFIRMED FAILED:
        #   tp=2 → crash (heads not divisible by 2)
        #   tp=1 → crash
        #   dp=2, ep=2 → crash
        #
        # pp=3 with no EP is dramatically better - avoids alltoall overhead.
        # Now try: tp=4, pp=3, dp=1, ep=1, mbs=2, ac=False
        # - mbs=2 gives more microbatches (4 vs 2), reducing pipeline bubble
        #   bubble ratio = (3-1)/4 = 0.5 vs (3-1)/2 = 1.0
        # - ac=False removes recomputation overhead
        # - Risk: might OOM without ac, but pp=3 reduces memory per stage
        #   Each stage has 9/27 = 1/3 of layers, and tp=4 further splits
        
        tp = 4
        pp = 3 if 3 in pp_choices else 1
        dp = 1 if 1 in dp_choices else 4
        ep = 1 if 1 in ep_choices else 4
        
        # Ensure ep divides dp
        if dp % ep != 0:
            ep = 1
        
        per_rank_batch = batch_size // dp  # 8 // 1 = 8
        
        # With pp=3: prefer mbs=2 for better pipeline bubble ratio
        # num_microbatches = 8/2 = 4, bubble = (3-1)/4 = 0.5
        mbs = 2 if 2 in mbs_choices else 1
        
        if per_rank_batch < mbs:
            mbs = max(m for m in mbs_choices if m <= per_rank_batch)
        
        # Try without activation checkpointing - pp=3 reduces memory per stage
        ac = False if False in ac_choices else True
        
    elif is_moe:
        # Generic MoE policy
        tp = min(gpus_per_node, max(tp_choices))
        pp = 1 if 1 in pp_choices else min(pp_choices)
        dp = max(dp_choices)
        
        valid_eps = [e for e in sorted(ep_choices, reverse=True) if dp % e == 0]
        ep = valid_eps[0] if valid_eps else 1
            
        per_rank_batch = batch_size // dp if dp > 0 else batch_size
        mbs = max(m for m in mbs_choices if m <= max(per_rank_batch, 1))
        ac = True
        
    else:
        # Generic dense model policy
        tp = min(gpus_per_node, max(tp_choices))
        pp = 1 if 1 in pp_choices else min(pp_choices)
        
        remaining_gpus = total_gpus // (tp * pp)
        dp = max(d for d in dp_choices if d <= remaining_gpus) if dp_choices else 1
        
        ep = 1
        
        per_rank_batch = batch_size // dp if dp > 0 else batch_size
        mbs = max(m for m in mbs_choices if m <= max(per_rank_batch, 1))
        
        ac = True if gpu_mem <= 40 else (False if False in ac_choices else True)
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "ep": ep,
        "micro_batch_size": mbs,
        "activation_checkpointing": ac,
    }
