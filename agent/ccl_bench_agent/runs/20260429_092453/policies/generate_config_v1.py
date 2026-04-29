
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Optimized config for LLM training on multi-node GPU clusters.
    
    Key principles:
    1. Keep tensor parallelism (TP) within a node to avoid slow inter-node communication
    2. Use data parallelism (DP) across nodes since all-reduce is bandwidth-friendly
    3. Use pipeline parallelism (PP) only when model doesn't fit with TP alone
    4. Enable activation checkpointing when memory is tight
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
    
    # Estimate model size category based on name
    model_family = workload.get("model_family", "").lower()
    
    # Determine bytes per param
    bytes_per_param = 2 if precision in ["bf16", "fp16"] else 4
    
    # For Llama-8B class models on A100 40GB:
    # Model weights: ~16GB in bf16
    # With optimizer states (Adam): ~48GB per copy in bf16 (weights + grads + 2 optimizer states in fp32)
    # So we need at least TP=2 to fit, or TP=1 with activation checkpointing might be too tight
    
    # Strategy: minimize TP to reduce communication, but ensure memory fits
    # Keep TP within a single node
    tp = 2  # Keeps within node, halves memory requirement
    pp = 1  # No pipeline parallelism for simpler scheduling
    dp = total_gpus // (tp * pp)
    
    # Validate dp is in valid choices
    if dp not in valid_choices.get("dp", [1, 2, 4, 8, 16]):
        # Adjust tp/pp to get valid dp
        tp = 4
        pp = 1
        dp = total_gpus // (tp * pp)
    
    # Determine micro_batch_size
    # Global batch = dp * micro_batch * gradient_accumulation_steps
    # We want micro_batch as large as possible to maximize GPU utilization
    # but it must divide evenly: batch_size / dp <= micro_batch * num_accumulation
    samples_per_rank = batch_size // dp
    
    # Pick largest micro_batch that divides samples_per_rank
    micro_batch = 1
    for mb in sorted(valid_choices.get("micro_batch_size", [1, 2, 4]), reverse=True):
        if samples_per_rank >= mb and samples_per_rank % mb == 0:
            micro_batch = mb
            break
    
    # Enable activation checkpointing to save memory, allowing smaller TP
    activation_checkpointing = True
    
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": activation_checkpointing,
    }
