
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
        # Confirmed working:
        #   tp=4, dp=4, pp=1, ep=4, mbs=1, ac=True → 3.359
        #   tp=4, dp=4, pp=1, ep=4, mbs=2, ac=True → 2.736 (best)
        #
        # Failed:
        #   tp=2 → crash (model heads not divisible by 2?)
        #   ep=2 with dp=4 → crash
        #
        # Next: Try dp=2, ep=1, mbs=4 to reduce communication overhead
        # With dp=2: per_rank_batch = 8/2 = 4, mbs=4 → 1 microbatch per step
        # Less allreduce (2-way vs 4-way), no alltoall (ep=1)
        # Trade: only 8 GPUs, more per-GPU memory/compute
        
        tp = 4 if 4 in tp_choices else max(tp_choices)
        pp = 1 if 1 in pp_choices else min(pp_choices)
        
        # Strategy: try dp=2 with larger mbs to reduce communication
        dp = 2 if 2 in dp_choices else 4
        
        # ep=1 to avoid alltoall entirely (ep=2 crashed before)
        ep = 1 if 1 in ep_choices else 4
        
        # With dp=2, per_rank_batch = 4, can use mbs=4
        per_rank_batch = batch_size // dp
        mbs = max(m for m in mbs_choices if m <= per_rank_batch)
        
        # Activation checkpointing needed on A100-40GB for MoE
        ac = True
        
    elif is_moe:
        # Generic MoE policy
        tp = min(gpus_per_node, max(tp_choices))
        pp = 1 if 1 in pp_choices else min(pp_choices)
        dp = max(dp_choices)
        
        # Moderate EP
        ep = min(e for e in sorted(ep_choices) if e > 1) if any(e > 1 for e in ep_choices) else 1
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
        
        ac = True if gpu_mem <= 40 else (True if True in ac_choices else False)
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "ep": ep,
        "micro_batch_size": mbs,
        "activation_checkpointing": ac,
    }
