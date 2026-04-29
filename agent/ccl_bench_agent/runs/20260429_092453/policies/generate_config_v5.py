
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Optimized config for LLM training on multi-node GPU clusters.
    
    History analysis:
    - tp=4, dp=4, pp=1, mbs=2, no ckpt: score 8.697 (SUCCESS)
    - tp=2, dp=8, pp=1, mbs=2, ckpt: FAILED (OOM)
    - tp=4, dp=4, pp=1, mbs=4, no ckpt: FAILED (OOM)
    - tp=4, dp=4, pp=1, mbs=4, ckpt: FAILED (error)
    - tp=2, pp=2, dp=4, mbs=2, no ckpt: score 6.524 (BEST)
    
    Key findings:
    - tp=2, pp=2 is the winning combination so far
    - Lower tp means less intra-node TP all-reduce communication
    - pp=2 splits model layers to fit with tp=2
    - mbs=2 with dp=4 gives 2 microbatches for 2 pipeline stages (50% bubble)
    
    New attempt: tp=2, pp=2, dp=4, mbs=1
    - mbs=1 gives 4 microbatches for 2 pipeline stages
    - More microbatches = smaller pipeline bubble fraction (2/(4+2-1) vs 1/(2+2-1))
    - Smaller microbatch = less memory per step
    - Trade-off: more kernel launches, but pipeline efficiency should improve
    """
    total_gpus = environment.get("total_gpus", 16)
    gpus_per_node = environment.get("gpus_per_node", 4)
    gpu_memory_gb = environment.get("gpu_memory_gb", 40)
    inter_bw = environment.get("inter_node_bandwidth_gbps", 25)
    intra_bw = environment.get("intra_node_bandwidth_gbps", 600)
    
    # Extract config space to know valid choices
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    batch_size = workload.get("batch_size", 16)
    seq_len = workload.get("seq_len", 1024)
    precision = workload.get("precision", "bf16")
    model_family = workload.get("model_family", "").lower()
    
    # Determine model size category
    is_large_model = any(x in model_family for x in ["70b", "65b", "40b", "34b"])
    is_medium_model = any(x in model_family for x in ["8b", "7b", "13b"])
    
    # Get valid choices
    tp_choices = valid_choices.get("tp", [1, 2, 4, 8])
    pp_choices = valid_choices.get("pp", [1, 2, 4, 8])
    dp_choices = valid_choices.get("dp", [1, 2, 4, 8, 16])
    mbs_choices = sorted(valid_choices.get("micro_batch_size", [1, 2, 4]))
    ckpt_choices = valid_choices.get("activation_checkpointing", [True, False])
    
    # Default strategy based on successful experiments
    if is_large_model:
        # Large models: need aggressive parallelism
        tp = min(gpus_per_node, max(tp_choices))
        pp = 4 if 4 in pp_choices else 2
        activation_checkpointing = True
    elif is_medium_model:
        # Medium models (8B): tp=2, pp=2 worked best
        # Try mbs=1 for better pipeline efficiency
        tp = 2
        pp = 2
        activation_checkpointing = False
    else:
        # Small models: minimal parallelism
        tp = 2
        pp = 1
        activation_checkpointing = False
    
    # Compute dp
    dp = total_gpus // (tp * pp)
    
    # Validate all choices
    if tp not in tp_choices:
        tp = max(c for c in tp_choices if c <= tp) if any(c <= tp for c in tp_choices) else min(tp_choices)
    if pp not in pp_choices:
        pp = max(c for c in pp_choices if c <= pp) if any(c <= pp for c in pp_choices) else min(pp_choices)
    
    dp = total_gpus // (tp * pp)
    if dp not in dp_choices:
        # Try adjusting - fall back to known working config
        tp = 4
        pp = 1
        dp = total_gpus // (tp * pp)
        if dp not in dp_choices:
            dp = min(dp_choices, key=lambda x: abs(x - total_gpus // (tp * pp)))
    
    # Determine micro_batch_size
    samples_per_rank = batch_size // dp if dp > 0 else batch_size
    
    # For pipeline parallelism, maximize number of microbatches to reduce bubble
    # bubble_fraction = (pp - 1) / (num_microbatches + pp - 1)
    if pp > 1:
        # Prefer smaller mbs to get more microbatches for pipeline efficiency
        micro_batch = 1  # Try mbs=1 for maximum pipeline efficiency
        if micro_batch not in mbs_choices:
            micro_batch = min(mbs_choices)
        # Verify it divides evenly
        if samples_per_rank % micro_batch != 0:
            # Find smallest valid mbs that divides samples_per_rank
            for mb in mbs_choices:
                if samples_per_rank >= mb and samples_per_rank % mb == 0:
                    micro_batch = mb
                    break
    else:
        # No pipeline: prefer larger mbs for better GPU utilization
        micro_batch = 1
        for mb in sorted(mbs_choices, reverse=True):
            if samples_per_rank >= mb and samples_per_rank % mb == 0:
                micro_batch = mb
                break
    
    # Memory safety check: if gpu_memory is tight, enable checkpointing
    if gpu_memory_gb <= 40 and is_medium_model and tp <= 2 and pp <= 1:
        activation_checkpointing = True
    
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": activation_checkpointing,
    }
