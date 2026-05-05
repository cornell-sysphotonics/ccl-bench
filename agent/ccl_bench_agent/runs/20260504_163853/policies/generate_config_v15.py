
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
        # Best known: tp=4, dp=1, pp=3, ep=1, mbs=4, ac=False → 1.595
        #
        # Key findings from 15 iterations:
        # - tp=2 fails (any combo)
        # - dp=2 fails (any combo)
        # - dp=4 works but slow (2.6-3.4) due to cross-node gradient sync
        # - ep>1 adds expensive alltoall → worse
        # - pp=3 is sweet spot: enough memory reduction without too much pipeline bubble
        # - pp=9 works but much slower (2.337 with ac=False)
        # - pp=1 with dp=1: only 4 GPUs, might OOM without PP, but worth trying with ac=True
        #   This eliminates pipeline bubble entirely
        # - mbs=4 >> mbs=2 >> mbs=1 (better GPU utilization)
        # - ac=False slightly faster than ac=True when memory allows
        
        # Try: tp=4, dp=1, pp=1, ep=1, mbs=4, ac=True
        # No pipeline bubble overhead. Only 4 GPUs (single node = fast comms).
        # ac=True to compensate for no PP memory reduction.
        # If this OOMs, fall back to pp=3 next iteration.
        
        tp = 4 if 4 in tp_choices else max(tp_choices)
        pp = 1 if 1 in pp_choices else min(pp_choices)
        dp = 1 if 1 in dp_choices else min(dp_choices)
        ep = 1 if 1 in ep_choices else min(ep_choices)
        
        # Ensure ep divides dp
        if dp % ep != 0:
            ep = 1
        
        # Ensure total GPUs fit
        total_used = tp * dp * pp
        if total_used > total_gpus:
            pp = 3 if 3 in pp_choices else min(p for p in pp_choices if p > 1)
            total_used = tp * dp * pp
        
        per_rank_batch = batch_size // dp
        
        # Use largest mbs for best GPU utilization
        mbs = max(m for m in mbs_choices if m <= per_rank_batch)
        
        # With pp=1, we need activation checkpointing to fit in memory
        if pp == 1:
            ac = True if True in ac_choices else False
        else:
            ac = False if False in ac_choices else True
        
    elif is_moe:
        # Generic MoE policy
        # Key principles:
        # 1. Keep TP within a node (tp <= gpus_per_node)
        # 2. Minimize EP to avoid expensive alltoall communication
        # 3. Use PP to split memory if model is large
        # 4. Maximize micro_batch_size for GPU utilization
        
        tp = min(gpus_per_node, max(tp_choices))
        
        # Use PP to reduce memory pressure on constrained GPUs
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
        
        if pp > 1:
            ac = False if False in ac_choices else True
        else:
            ac = True if True in ac_choices else False
        
    else:
        # Generic dense model policy
        tp = min(gpus_per_node, max(tp_choices))
        
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
