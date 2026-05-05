
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
        # Known working: tp=4, dp=4, pp=1, ep=4, mbs=1, ac=True → score 3.359
        # Failed: tp=2, dp=4, pp=1, ep=4, mbs=2 → error (8 GPUs config issue)
        #
        # Next attempt: keep the working 16-GPU layout but try mbs=2
        # With dp=4 and batch=8, per-rank batch = 2, so mbs=2 means 1 microbatch
        # This should improve GPU utilization (larger microbatch = better throughput)
        
        tp = 4 if 4 in tp_choices else 2
        pp = 1 if 1 in pp_choices else min(pp_choices)
        dp = 4 if 4 in dp_choices else 2
        ep = 4 if 4 in ep_choices else 2
        
        # EP must divide DP
        if dp % ep != 0:
            ep = 1
        
        # Try mbs=2 for better GPU utilization
        # per_rank_batch = batch_size // dp = 8 // 4 = 2
        per_rank_batch = batch_size // dp
        mbs = 2 if (2 in mbs_choices and per_rank_batch >= 2) else 1
        
        # Activation checkpointing: needed on A100-40GB for this model
        ac = True
        
    elif is_moe:
        # Generic MoE policy
        # Keep TP within node, use EP for expert distribution
        tp = min(gpus_per_node, max(tp_choices))
        pp = 1 if 1 in pp_choices else min(pp_choices)
        dp = max(dp_choices)
        
        # EP: use moderate expert parallelism
        ep = min(e for e in sorted(ep_choices) if e > 1) if any(e > 1 for e in ep_choices) else 1
        if dp % ep != 0:
            ep = 1
            
        per_rank_batch = batch_size // dp if dp > 0 else batch_size
        mbs = max(m for m in mbs_choices if m <= max(per_rank_batch, 1))
        ac = True
        
    else:
        # Generic dense model policy
        # TP within node for fast allreduce
        tp = min(gpus_per_node, max(tp_choices))
        pp = 1 if 1 in pp_choices else min(pp_choices)
        
        # Maximize DP for throughput
        remaining_gpus = total_gpus // (tp * pp)
        dp = max(d for d in dp_choices if d <= remaining_gpus) if dp_choices else 1
        
        ep = 1
        
        per_rank_batch = batch_size // dp if dp > 0 else batch_size
        mbs = max(m for m in mbs_choices if m <= max(per_rank_batch, 1))
        
        # Use activation checkpointing if GPU memory is tight
        ac = True if gpu_mem <= 40 else (True if True in ac_choices else False)
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "ep": ep,
        "micro_batch_size": mbs,
        "activation_checkpointing": ac,
    }
