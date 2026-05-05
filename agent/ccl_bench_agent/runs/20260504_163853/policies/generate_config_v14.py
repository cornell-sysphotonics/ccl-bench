
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
        # Empirically determined best config: tp=4, dp=1, pp=3, ep=1, mbs=4, ac=False → 1.595
        #
        # Key findings from 14 iterations:
        # - tp=2 fails for this model (any combo)
        # - dp=2 fails for this model  
        # - pp=1 with dp=1 OOMs (need pp>=3 to fit in 40GB)
        # - ep=1 >> ep=4 (alltoall overhead dominates with ep>1)
        # - mbs=4 >> mbs=2 >> mbs=1 (larger mbs = better GPU util)
        # - ac=False slightly better than ac=True when memory allows
        
        tp = 4 if 4 in tp_choices else max(tp_choices)
        pp = 3 if 3 in pp_choices else min(p for p in pp_choices if p > 1) if any(p > 1 for p in pp_choices) else 1
        dp = 1 if 1 in dp_choices else min(dp_choices)
        ep = 1 if 1 in ep_choices else min(ep_choices)
        
        # Ensure ep divides dp
        if dp % ep != 0:
            ep = 1
        
        # Ensure total GPUs fit
        total_used = tp * dp * pp
        if total_used > total_gpus:
            # Fall back to smaller config
            pp = 1
            total_used = tp * dp * pp
        
        per_rank_batch = batch_size // dp
        
        # Use largest mbs for best GPU utilization
        mbs = max(m for m in mbs_choices if m <= per_rank_batch)
        
        # No activation checkpointing for speed (memory fits with pp=3)
        ac = False if False in ac_choices else True
        
    elif is_moe:
        # Generic MoE policy
        # Key principles:
        # 1. Keep TP within a node (tp <= gpus_per_node)
        # 2. Minimize EP to avoid expensive alltoall communication
        # 3. Use PP to split memory if model is large
        # 4. Maximize micro_batch_size for GPU utilization
        
        tp = min(gpus_per_node, max(tp_choices))
        
        # Use PP to reduce memory pressure
        # Prefer smallest PP > 1 to balance memory vs pipeline bubble
        pp_candidates = [p for p in sorted(pp_choices) if p > 1 and tp * p <= total_gpus]
        if pp_candidates and gpu_mem <= 40:
            pp = pp_candidates[0]
        else:
            pp = 1
        
        remaining_gpus = total_gpus // (tp * pp)
        dp = max(d for d in dp_choices if d <= remaining_gpus) if dp_choices else 1
        
        # Minimize EP - alltoall is expensive especially cross-node
        ep = 1 if 1 in ep_choices else min(ep_choices)
        if dp % ep != 0:
            ep = 1
            
        per_rank_batch = batch_size // dp if dp > 0 else batch_size
        mbs = max(m for m in mbs_choices if m <= max(per_rank_batch, 1))
        
        # Prefer no activation checkpointing if using PP to save memory
        if pp > 1:
            ac = False if False in ac_choices else True
        else:
            ac = True if True in ac_choices else False
        
    else:
        # Generic dense model policy
        # 1. Maximize TP within node for memory reduction
        # 2. Use PP if model is large and memory-constrained
        # 3. Use DP for throughput scaling
        
        tp = min(gpus_per_node, max(tp_choices))
        
        # For large models on limited GPU memory, use PP
        if gpu_mem <= 40 and num_layers and num_layers > 32:
            pp_candidates = [p for p in sorted(pp_choices) if p > 1 and tp * p <= total_gpus]
            pp = pp_candidates[0] if pp_candidates else 1
        else:
            pp = 1 if 1 in pp_choices else min(pp_choices)
        
        remaining_gpus = total_gpus // (tp * pp)
        dp = max(d for d in dp_choices if d <= remaining_gpus) if dp_choices else 1
        
        ep = 1
        
        per_rank_batch = batch_size // dp if dp > 0 else batch_size
        mbs = max(m for m in mbs_choices if m <= max(per_rank_batch, 1))
        
        # Activation checkpointing on smaller GPUs
        if gpu_mem <= 40:
            ac = True if True in ac_choices else False
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
