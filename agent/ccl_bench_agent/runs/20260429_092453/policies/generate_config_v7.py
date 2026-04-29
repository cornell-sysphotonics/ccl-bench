
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Optimized config for LLM training on multi-node GPU clusters.
    
    History analysis:
    - tp=4, dp=4, pp=1, mbs=2, no ckpt: score 8.697
    - tp=2, pp=2, dp=4, mbs=2, no ckpt: score 6.524
    - tp=2, pp=2, dp=4, mbs=1, no ckpt: score 6.061 (BEST)
    - tp=2, pp=1, dp=8: FAILED (OOM)
    - tp=1, pp=2, dp=8, ckpt=True: FAILED
    - tp=4, pp=1, dp=4, mbs=4: FAILED (OOM)
    
    Lessons:
    - pp=2 with tp=2 works well, mbs=1 is better than mbs=2
    - tp=1 doesn't work for 8B model on 40GB A100
    - No checkpointing is faster when memory allows
    
    New attempt: tp=4, pp=2, dp=2, mbs=1, no ckpt
    - tp=4 within a node (4 GPUs/node), reduces memory per GPU significantly
    - pp=2 splits model into 2 stages
    - dp=2 means 8 samples per rank, 8 microbatches with mbs=1
    - Pipeline bubble = (pp-1)/(num_microbatches+pp-1) = 1/9 ≈ 11% (vs 20% for best)
    - Lower bubble fraction may compensate for higher TP communication
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
    
    # Strategy: tp=4 within node, pp=2 for memory, dp=2 for low bubble
    tp = 4
    pp = 2
    activation_checkpointing = False
    
    # Compute dp
    dp = total_gpus // (tp * pp)
    
    # Validate all choices
    if tp not in tp_choices:
        tp = 2
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
    
    # Use mbs=1 for best pipeline efficiency
    micro_batch = 1
    if micro_batch not in mbs_choices:
        micro_batch = min(mbs_choices)
    
    # Verify it divides evenly
    if samples_per_rank % micro_batch != 0:
        for mb in mbs_choices:
            if samples_per_rank >= mb and samples_per_rank % mb == 0:
                micro_batch = mb
                break
    
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": activation_checkpointing,
    }
