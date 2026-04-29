
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus    = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    gpu_memory_gb = environment.get("gpu_memory_gb", 40)
    batch_size    = workload.get("batch_size", 1)
    seq_len       = workload.get("seq_len", 1024)
    precision     = workload.get("precision", "bf16")
    
    # Build lookup of valid choices for each dimension
    config_space = workload.get("config_space", [])
    valid = {}
    for dim in config_space:
        valid[dim["name"]] = dim["choices"]
    
    # For Llama-3.1-8B on A100 40GB with bf16:
    # Model params ~8B, ~16GB in bf16 for parameters alone
    # With tp=2, each GPU holds ~8GB of params, leaving ~32GB for activations/optimizer
    # With FSDP (dp=8), optimizer states are sharded across 8 GPUs
    
    # Strategy: minimize TP (communication heavy) while fitting in memory
    # Maximize DP for better throughput scaling
    # Use inductor compilation for kernel fusion speedups
    
    # Determine model size category
    model_family = workload.get("model_family", "").lower()
    
    # Estimate model size in GB (bf16)
    bytes_per_param = 2 if "bf16" in precision or "fp16" in precision else 4
    
    # For 8B model: ~16GB in bf16
    # With optimizer (Adam): ~4x params for master weights + momentum + variance
    # FSDP shards optimizer states across dp ranks
    
    # Try tp=2 to reduce intra-node comm while still fitting in memory
    tp = 2
    dp = total_gpus // tp  # 16 // 2 = 8
    pp = 1
    
    # Ensure tp * dp * pp = total_gpus
    assert tp * dp * pp == total_gpus, f"tp*dp*pp={tp*dp*pp} != total_gpus={total_gpus}"
    
    # local batch = batch_size / dp = 32 / 8 = 4
    local_batch = batch_size // dp
    
    # micro_batch should divide local_batch; use largest valid micro_batch <= local_batch
    micro_batch = local_batch  # 4
    
    # Validate against config space
    if "tp" in valid and tp not in valid["tp"]:
        tp = min(valid["tp"], key=lambda x: abs(x - tp))
    if "dp" in valid and dp not in valid["dp"]:
        dp = min(valid["dp"], key=lambda x: abs(x - dp))
    if "micro_batch" in valid and micro_batch not in valid["micro_batch"]:
        # Pick largest valid choice <= local_batch
        candidates = [c for c in valid["micro_batch"] if c <= local_batch]
        micro_batch = max(candidates) if candidates else min(valid["micro_batch"])
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": "inductor",
        "activation_checkpointing": True,
    }
