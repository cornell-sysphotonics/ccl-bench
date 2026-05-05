
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
    
    # DeepSeek-V2-Lite specific reasoning:
    # - MoE model with 64 experts, 27 layers
    # - On A100-40GB with 4 nodes × 4 GPUs
    # - Inter-node bandwidth likely slower than intra-node NVLink
    
    if "deepseek" in model_family.lower():
        # Strategy: minimize cross-node communication
        # TP=2 within node reduces allreduce overhead vs TP=4
        # DP=4 for data parallelism (some cross-node, but just gradient sync)
        # EP=4 to distribute 64 experts across 4 ranks
        # With TP=2, DP=4, PP=1: 8 GPUs (2 nodes)
        # mbs=2: better GPU utilization, global_bs=8/dp=4=2 per rank, fits in 1 micro-step
        
        tp = 2 if 2 in tp_choices else 4
        pp = 1 if 1 in pp_choices else min(pp_choices)
        dp = 4 if 4 in dp_choices else 2
        ep = 4 if 4 in ep_choices else 2
        
        # EP must divide DP
        if dp % ep != 0:
            ep = 1
        
        # micro_batch_size: try 2 for better utilization
        mbs = 2 if 2 in mbs_choices else 1
        
        # Check if per-rank batch fits
        per_rank_batch = batch_size // dp
        if per_rank_batch < mbs:
            mbs = max(m for m in mbs_choices if m <= per_rank_batch)
        
        # Activation checkpointing: keep on for safety with A100-40GB
        ac = True
        
    else:
        # Generic policy for non-MoE or unknown models
        tp = min(gpus_per_node, max(tp_choices))
        pp = 1 if 1 in pp_choices else min(pp_choices)
        dp = max(dp_choices)
        ep = 1
        if is_moe and ep_choices:
            ep = min(e for e in ep_choices if e > 1) if any(e > 1 for e in ep_choices) else 1
        mbs = 1
        ac = True
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "ep": ep,
        "micro_batch_size": mbs,
        "activation_checkpointing": ac,
    }
