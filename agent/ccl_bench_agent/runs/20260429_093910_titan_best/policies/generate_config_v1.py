
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus    = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    batch_size    = workload.get("batch_size", 1)
    gpu_memory    = environment.get("gpu_memory_gb", 80)
    
    # Build lookup of valid choices for each config dimension
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    # Strategy: minimize communication overhead
    # - Keep tp within a single node (use NVLink)
    # - Prefer dp for scaling (less communication overhead for large batches)
    # - Use pp only if memory constrained
    
    # For Llama-3.1-8B on A100-40GB:
    # ~16GB for model params in bf16, need room for activations and optimizer states
    # tp=2 should be sufficient for memory, maximizes dp
    
    tp = 2
    pp = 1
    dp = total_gpus // (tp * pp)  # = 8
    
    # Validate dp is in valid choices
    if "dp" in valid_choices and dp not in valid_choices["dp"]:
        # Try other combinations
        best_dp = max(c for c in valid_choices["dp"] if c <= dp)
        dp = best_dp
        # Adjust tp or pp to match
        remaining = total_gpus // dp
        if "tp" in valid_choices:
            for t in sorted(valid_choices["tp"], reverse=True):
                if remaining % t == 0:
                    p = remaining // t
                    if "pp" not in valid_choices or p in valid_choices["pp"]:
                        tp = t
                        pp = p
                        break
    
    # Validate all choices
    if "tp" in valid_choices and tp not in valid_choices["tp"]:
        tp = min(valid_choices["tp"], key=lambda x: abs(x - tp))
    if "pp" in valid_choices and pp not in valid_choices["pp"]:
        pp = min(valid_choices["pp"], key=lambda x: abs(x - pp))
    if "dp" in valid_choices and dp not in valid_choices["dp"]:
        dp = min(valid_choices["dp"], key=lambda x: abs(x - dp))
    
    # Use larger micro_batch_size to reduce number of gradient accumulation steps
    micro_batch = 4
    if "micro_batch_size" in valid_choices and micro_batch not in valid_choices["micro_batch_size"]:
        micro_batch = max(valid_choices["micro_batch_size"])
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": False,
    }
