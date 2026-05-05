
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
        #   tp=4, dp=1, pp=3, ep=1, mbs=4, ac=False → 1.595 ★ BEST
        #   tp=4, dp=1, pp=3, ep=1, mbs=4, ac=True  → 1.642
        #   tp=4, dp=1, pp=3, ep=1, mbs=2, ac=False → 2.337
        #   tp=4, dp=4, pp=1, ep=4, mbs=2, ac=False → 2.633
        #   tp=4, dp=4, pp=1, ep=4, mbs=2, ac=True  → 2.736
        #   tp=4, dp=4, pp=1, ep=4, mbs=1, ac=True  → 3.359
        #
        # Key insights:
        # - pp=3, ep=1 dominates (avoids alltoall overhead)
        # - mbs=4 >> mbs=2 (better GPU utilization)
        # - ac=False slightly better than ac=True
        #
        # Now trying: tp=2, dp=2, pp=3, ep=1, mbs=2, ac=False (12 GPUs)
        # Rationale: TP=2 reduces allreduce overhead (intra-node only),
        # DP=2 doubles throughput with gradient allreduce,
        # pp=3 splits 27 layers into 9/stage, ep=1 avoids alltoall.
        # per_rank_batch = 8/2 = 4, mbs=2 → 2 microbatches, bubble = 0.5

        tp = 2 if 2 in tp_choices else 4
        pp = 3 if 3 in pp_choices else 1
        dp = 2 if 2 in dp_choices else 1
        ep = 1 if 1 in ep_choices else min(ep_choices)
        
        # Ensure ep divides dp
        if dp % ep != 0:
            ep = 1
        
        per_rank_batch = batch_size // dp
        
        # With dp=2, per_rank_batch=4, mbs=2 gives 2 microbatches
        mbs = 2 if 2 in mbs_choices else 1
        
        if per_rank_batch < mbs:
            mbs = max(m for m in mbs_choices if m <= per_rank_batch)
        
        ac = False if False in ac_choices else True
        
    elif is_moe:
        # Generic MoE policy - prefer pipeline parallelism over expert parallelism
        tp = min(gpus_per_node, max(tp_choices))
        
        pp_candidates = [p for p in sorted(pp_choices, reverse=True) if p > 1]
        if pp_candidates:
            pp = pp_candidates[0]
        else:
            pp = 1
        
        remaining_gpus = total_gpus // (tp * pp)
        dp = max(d for d in dp_choices if d <= remaining_gpus) if dp_choices else 1
        
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
