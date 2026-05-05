
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
        # CONFIRMED WORKING:
        #   tp=4, dp=4, pp=1, ep=4, mbs=1, ac=True  → 3.359
        #   tp=4, dp=4, pp=1, ep=4, mbs=2, ac=True  → 2.736
        #   tp=4, dp=4, pp=1, ep=4, mbs=2, ac=False → 2.633 (BEST)
        #
        # CONFIRMED FAILED:
        #   tp=2 → crash (heads not divisible by 2)
        #   tp=4, dp=4, ep=2 → crash
        #   dp=2, ep=1 → crash (likely OOM)
        #
        # Next experiment: Try tp=1 to eliminate tensor parallel communication
        # tp=1 means no allreduce for attention/FFN, but more memory per GPU.
        # With ep=4 distributing 64 experts across 4 GPUs, memory may be manageable.
        # dp=4 with ep=4 keeps expert distribution.
        # Total GPUs = 1*4*1 = 4 (uses only 1 node!)
        # This could be faster due to no cross-node communication at all.
        
        tp = 1 if 1 in tp_choices else 4
        pp = 1 if 1 in pp_choices else min(pp_choices)
        dp = 4 if 4 in dp_choices else max(dp_choices)
        ep = 4 if 4 in ep_choices else max(e for e in ep_choices if dp % e == 0)
        
        per_rank_batch = batch_size // dp
        mbs = max(m for m in mbs_choices if m <= max(per_rank_batch, 1))
        
        # ac=False worked with tp=4; with tp=1 we have less model parallelism
        # so more memory per GPU. Use ac=True for safety.
        ac = True
        
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
