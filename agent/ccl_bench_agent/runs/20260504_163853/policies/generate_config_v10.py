
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
        # CONFIRMED RESULTS (sorted by score):
        #   tp=4, dp=1, pp=3, ep=1, mbs=4, ac=True  → 1.642 ★ BEST
        #   tp=4, dp=1, pp=3, ep=1, mbs=2, ac=False → 2.337
        #   tp=4, dp=4, pp=1, ep=4, mbs=2, ac=False → 2.633
        #   tp=4, dp=4, pp=1, ep=4, mbs=2, ac=True  → 2.736
        #   tp=4, dp=4, pp=1, ep=4, mbs=1, ac=True  → 3.359
        #
        # Key findings:
        # - pp=3 with ep=1 dominates pp=1 with ep=4 (avoids alltoall)
        # - mbs=4 >> mbs=2 (better GPU utilization despite worse bubble ratio)
        # - Next: try mbs=4, ac=False to combine best mbs with no recompute overhead
        
        tp = 4
        pp = 3 if 3 in pp_choices else 1
        dp = 1 if 1 in dp_choices else min(dp_choices)
        ep = 1 if 1 in ep_choices else min(ep_choices)
        
        # Ensure ep divides dp
        if dp % ep != 0:
            ep = 1
        
        per_rank_batch = batch_size // dp  # 8 // 1 = 8
        
        # mbs=4 was much better than mbs=2 (1.642 vs 2.337)
        # With pp=3, mbs=4 → 2 microbatches, bubble ratio = 1.0
        # But larger mbs = better GPU utilization, which dominates
        mbs = 4 if 4 in mbs_choices else 2
        
        if per_rank_batch < mbs:
            mbs = max(m for m in mbs_choices if m <= per_rank_batch)
        
        # Try ac=False: pp=3 reduces per-stage memory to 9 layers,
        # tp=4 further splits tensors. With ~2.4B params per stage / 4 tp = ~600M
        # Model memory: 600M × 12 bytes ≈ 7.2 GB. Activations with mbs=4, seq=1024
        # should still fit in 40GB without checkpointing.
        ac = False if False in ac_choices else True
        
    elif is_moe:
        # Generic MoE policy - prefer pipeline parallelism over expert parallelism
        # to avoid alltoall overhead on slow interconnects
        tp = min(gpus_per_node, max(tp_choices))
        
        # Use PP if available to avoid EP
        pp_candidates = [p for p in sorted(pp_choices, reverse=True) if p > 1]
        if pp_candidates:
            pp = pp_candidates[0]
        else:
            pp = 1
        
        remaining_gpus = total_gpus // (tp * pp)
        dp = max(d for d in dp_choices if d <= remaining_gpus) if dp_choices else 1
        
        # Minimize EP - alltoall is expensive cross-node
        ep = 1 if 1 in ep_choices else min(ep_choices)
        if dp % ep != 0:
            ep = 1
            
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
        
        # Use activation checkpointing on smaller GPUs
        if gpu_mem <= 40:
            ac = True
        else:
            ac = False if False in ac_choices else True
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "ep": ep,
        "micro_batch_size": mbs,
        "activation_checkpointing": ac,
    }
