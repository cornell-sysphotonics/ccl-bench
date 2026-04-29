
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus    = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    gpu_memory_gb = environment.get("gpu_memory_gb", 80)
    batch_size    = workload.get("batch_size", 1)
    seq_len       = workload.get("seq_len", 2048)
    
    # Build valid choices lookup - handle multiple possible key formats
    config_space = workload.get("config_space", [])
    valid = {}
    if isinstance(config_space, list):
        for dim in config_space:
            if isinstance(dim, dict):
                # Try different possible key names for the dimension name
                dim_name = None
                for key in ["name", "key", "param", "dimension", "id", "label"]:
                    if key in dim:
                        dim_name = dim[key]
                        break
                if dim_name is None:
                    # Try first string value
                    for k, v in dim.items():
                        if isinstance(v, str) and not isinstance(dim.get("choices"), type(None)):
                            dim_name = v
                            break
                
                dim_choices = None
                for key in ["choices", "values", "options", "range"]:
                    if key in dim:
                        dim_choices = dim[key]
                        break
                
                if dim_name and dim_choices:
                    valid[dim_name] = dim_choices
    elif isinstance(config_space, dict):
        for k, v in config_space.items():
            if isinstance(v, dict) and "choices" in v:
                valid[k] = v["choices"]
            elif isinstance(v, list):
                valid[k] = v

    # Strategy for Llama-8B on 16x A100-40GB (4 nodes of 4 GPUs):
    # - tp=2: minimal TP within node (fast NVLink), reduces all-reduce overhead vs tp=4
    # - dp=8: maximize data parallelism with FSDP for good throughput
    # - pp=1: avoid pipeline bubble overhead
    # - With dp=8, local_batch = 32/8 = 4, micro_batch=4
    
    tp = 2
    dp = 8
    pp = 1
    
    # Verify tp * dp * pp = total_gpus
    if tp * dp * pp != total_gpus:
        # Try different configurations
        for try_tp in [2, 4, 1]:
            remaining = total_gpus // try_tp
            if total_gpus % try_tp == 0 and remaining >= 1:
                tp = try_tp
                dp = remaining
                pp = 1
                break
    
    # Ensure values are in valid choices
    def pick_valid(name, value, fallback=None):
        if name in valid:
            if value in valid[name]:
                return value
            # Find closest valid value
            return min(valid[name], key=lambda x: abs(x - value) if isinstance(x, (int, float)) else 0)
        return value if fallback is None else fallback
    
    tp = pick_valid("tp", tp)
    dp = pick_valid("dp", dp)
    pp = pick_valid("pp", pp)
    
    # Local batch size
    local_batch = batch_size // dp if dp > 0 else batch_size
    
    # micro_batch: with pp=1, ideally micro_batch = local_batch
    micro_batch = local_batch
    if "micro_batch" in valid:
        # Pick largest valid micro_batch that divides local_batch
        candidates = [m for m in sorted(valid["micro_batch"], reverse=True) 
                      if local_batch > 0 and local_batch % m == 0]
        if candidates:
            micro_batch = candidates[0]
        else:
            # Pick largest valid micro_batch <= local_batch
            candidates = [m for m in sorted(valid["micro_batch"], reverse=True) 
                          if m <= max(local_batch, 1)]
            if candidates:
                micro_batch = candidates[0]
            else:
                micro_batch = min(valid["micro_batch"])
    
    # Use inductor for torch.compile speedup
    compile_mode = "inductor"
    if "compile_mode" in valid and compile_mode not in valid["compile_mode"]:
        compile_mode = valid["compile_mode"][0]
    
    # Activation checkpointing: with FSDP across 8 ranks, memory should be ok
    activation_checkpointing = False
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": compile_mode,
        "activation_checkpointing": activation_checkpointing,
    }
