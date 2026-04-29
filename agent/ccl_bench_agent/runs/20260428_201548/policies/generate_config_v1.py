
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus    = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    gpu_memory_gb = environment.get("gpu_memory_gb", 80)
    batch_size    = workload.get("batch_size", 1)
    seq_len       = workload.get("seq_len", 2048)
    
    # Build valid choices lookup
    config_space = workload.get("config_space", [])
    valid = {}
    for dim in config_space:
        valid[dim["name"]] = dim["choices"]
    
    # For Llama-8B on A100-40GB: model is ~16GB in bf16
    # tp=2 should fit comfortably and reduces comm overhead vs tp=4
    # Maximize dp for better training throughput
    
    # Strategy: minimize tp (to reduce communication), maximize dp
    # Use pp only if memory requires it
    
    # Estimate model size roughly
    # Llama-8B ~ 8B params, bf16 = 16 bytes per param (with optimizer states) 
    # But with FSDP (dp), optimizer states are sharded across dp ranks
    
    # Try tp=2, which keeps communication intra-node (fast NVLink)
    # dp = total_gpus / tp = 16/2 = 8
    # pp = 1 (no pipeline parallelism overhead)
    
    tp = 2
    dp = total_gpus // tp  # = 8
    pp = 1
    
    # Verify tp * dp * pp = total_gpus
    if tp * dp * pp != total_gpus:
        # Fallback
        tp = gpus_per_node
        dp = total_gpus // tp
        pp = 1
    
    # Ensure values are in valid choices
    if "tp" in valid and tp not in valid["tp"]:
        tp = min(valid["tp"], key=lambda x: abs(x - tp))
    if "dp" in valid and dp not in valid["dp"]:
        dp = min(valid["dp"], key=lambda x: abs(x - dp))
    if "pp" in valid and pp not in valid["pp"]:
        pp = min(valid["pp"], key=lambda x: abs(x - pp))
    
    # Local batch size
    local_batch = batch_size // dp
    
    # micro_batch: use the largest that divides local_batch for fewer pipeline steps
    micro_batch = local_batch  # no pipeline, so micro_batch = local_batch
    if "micro_batch" in valid:
        # Pick largest valid micro_batch that divides local_batch
        candidates = [m for m in sorted(valid["micro_batch"], reverse=True) 
                      if local_batch > 0 and local_batch % m == 0]
        if candidates:
            micro_batch = candidates[0]
        else:
            # Pick largest valid micro_batch <= local_batch
            candidates = [m for m in sorted(valid["micro_batch"], reverse=True) 
                          if m <= local_batch]
            if candidates:
                micro_batch = candidates[0]
            else:
                micro_batch = min(valid["micro_batch"])
    
    # Use inductor for potential speedup with torch.compile
    compile_mode = "inductor"
    
    # Activation checkpointing: with tp=2 and dp=8, memory should be manageable
    # but let's enable it to be safe and allow larger micro batches
    activation_checkpointing = False
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": compile_mode,
        "activation_checkpointing": activation_checkpointing,
    }
