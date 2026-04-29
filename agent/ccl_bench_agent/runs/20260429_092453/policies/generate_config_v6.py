
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Optimized config for LLM training on multi-node GPU clusters.
    
    History analysis:
    - tp=4, dp=4, pp=1, mbs=2, no ckpt: score 8.697
    - tp=2, dp=8, pp=1, mbs=2, ckpt: FAILED (tp=2 pp=1 doesn't split model enough)
    - tp=4, dp=4, pp=1, mbs=4, no ckpt: FAILED (OOM, mbs=4 too large)
    - tp=4, dp=4, pp=1, mbs=4, ckpt: FAILED
    - tp=2, pp=2, dp=4, mbs=2, no ckpt: score 6.524
    - tp=2, pp=2, dp=4, mbs=1, no ckpt: score 6.061 (BEST)
    
    New attempt: tp=1, pp=2, dp=8, mbs=1, no ckpt
    - Eliminates TP all-reduce communication entirely
    - pp=2 splits model to fit in memory (half layers per stage)
    - dp=8 maximizes data parallelism
    - mbs=1 for pipeline efficiency
    - Risk: tp=1 means each GPU holds full layer weights for its pipeline stage
      For 8B model with pp=2: ~8GB weights + ~12GB optimizer per stage = ~20GB, 
      plus activations. Should fit in 40GB A100.
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
    
    # Strategy: try tp=1, pp=2, dp=8 to eliminate TP communication
    tp = 1
    pp = 2
    activation_checkpointing = False
    
    # Compute dp
    dp = total_gpus // (tp * pp)
    
    # Validate all choices
    if tp not in tp_choices:
        tp = min(c for c in tp_choices if c >= 1) if 1 in tp_choices else 2
    if pp not in pp_choices:
        pp = 2 if 2 in pp_choices else min(pp_choices)
    
    dp = total_gpus // (tp * pp)
    if dp not in dp_choices:
        # Fall back to known best: tp=2, pp=2, dp=4
        tp = 2
        pp = 2
        dp = total_gpus // (tp * pp)
        if dp not in dp_choices:
            dp = min(dp_choices, key=lambda x: abs(x - total_gpus // (tp * pp)))
    
    # Determine micro_batch_size
    samples_per_rank = batch_size // dp if dp > 0 else batch_size
    
    # For pipeline parallelism, prefer smaller mbs for more microbatches (less bubble)
    micro_batch = 1
    if micro_batch not in mbs_choices:
        micro_batch = min(mbs_choices)
    
    # Verify it divides evenly
    if samples_per_rank % micro_batch != 0:
        for mb in mbs_choices:
            if samples_per_rank >= mb and samples_per_rank % mb == 0:
                micro_batch = mb
                break
    
    # Memory safety: for tp=1 on medium models, might need checkpointing
    # 8B model, pp=2: ~4B params per stage, bf16 weights ~8GB, optimizer ~24GB total
    # With seq=1024, batch activations could be large
    # Let's keep checkpointing off but be ready to enable if needed
    if tp == 1 and is_medium_model and gpu_memory_gb <= 40:
        # Activations for 8B/2 stages at seq=1024 might be tight without checkpointing
        # Enable checkpointing to be safe
        activation_checkpointing = True
    
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": activation_checkpointing,
    }
