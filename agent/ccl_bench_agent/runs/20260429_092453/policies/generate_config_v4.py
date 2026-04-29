
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Optimized config for LLM training on multi-node GPU clusters.
    
    History:
    - tp=4, dp=4, pp=1, mbs=2, no checkpoint: score 8.697 (SUCCESS)
    - tp=2, dp=8, pp=1, mbs=2, checkpoint: FAILED (OOM - tp=2 not enough for 8B on 40GB)
    - tp=4, dp=4, pp=1, mbs=4, no checkpoint: FAILED (OOM)
    - tp=4, dp=4, pp=1, mbs=4, checkpoint: FAILED (error)
    
    New attempt: tp=2, pp=2, dp=4, mbs=2, no checkpoint
    - pp=2 splits layers across 2 stages, halving per-stage memory
    - This should allow tp=2 to fit (unlike pp=1 where tp=2 OOM'd)
    - tp=2 has less TP communication overhead than tp=4
    - dp=4 keeps good data parallelism
    - Pipeline with 2 microbatches (16/4/2 = 2 grad_accum steps)
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
    
    # For this specific scenario: try tp=2, pp=2, dp=4
    # pp=2 halves per-stage layers, so tp=2 should now fit for 8B model
    # Less TP comm than tp=4, but adds pipeline bubble overhead
    tp = 2
    pp = 2
    dp = total_gpus // (tp * pp)  # = 4
    
    # Validate choices
    tp_choices = valid_choices.get("tp", [1, 2, 4, 8])
    pp_choices = valid_choices.get("pp", [1, 2, 4, 8])
    dp_choices = valid_choices.get("dp", [1, 2, 4, 8, 16])
    
    if tp not in tp_choices or pp not in pp_choices or dp not in dp_choices:
        # Fallback to known working config
        tp = 4
        pp = 1
        dp = total_gpus // (tp * pp)
    
    # For large models, use more aggressive parallelism
    if is_large_model:
        tp = min(8, gpus_per_node)
        pp = 2
        activation_checkpointing = True
        if tp not in tp_choices:
            tp = max(c for c in tp_choices if c <= tp)
        if pp not in pp_choices:
            pp = max(c for c in pp_choices if c <= pp)
        dp = total_gpus // (tp * pp)
        if dp not in dp_choices:
            tp = 4
            pp = 1
            dp = total_gpus // (tp * pp)
    
    # No activation checkpointing to avoid recomputation overhead
    activation_checkpointing = False
    
    # Determine micro_batch_size
    mbs_choices = sorted(valid_choices.get("micro_batch_size", [1, 2, 4]), reverse=True)
    
    samples_per_rank = batch_size // dp if dp > 0 else batch_size
    
    # For pipeline parallelism, we want enough microbatches to fill the pipeline
    # num_microbatches = samples_per_rank / mbs >= pp for good efficiency
    micro_batch = 1
    for mb in mbs_choices:
        if samples_per_rank >= mb and samples_per_rank % mb == 0:
            num_microbatches = samples_per_rank // mb
            # For pp > 1, prefer more microbatches to reduce bubble
            if pp > 1 and num_microbatches < pp:
                continue
            micro_batch = mb
            break
    
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": activation_checkpointing,
    }
