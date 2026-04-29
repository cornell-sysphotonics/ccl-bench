
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    gpu_memory_gb = environment.get("gpu_memory_gb", 80)
    batch_size = workload.get("batch_size", 1)
    seq_len = workload.get("seq_len", 1024)
    model_family = workload.get("model_family", "").lower()
    intra_bw = environment.get("intra_node_bandwidth_gbps", 600)
    inter_bw = environment.get("inter_node_bandwidth_gbps", 100)
    
    # Build a lookup of valid choices for each config key
    config_space = workload.get("config_space", [])
    valid = {}
    for dim in config_space:
        valid[dim["key"]] = dim["choices"]
    
    # Helper to pick closest valid choice
    def pick_valid(key, target):
        choices = valid.get(key, [target])
        if all(isinstance(c, (int, float)) for c in choices):
            return min(choices, key=lambda c: abs(c - target))
        if target in choices:
            return target
        return choices[0]
    
    # Estimate model size in billions of parameters
    model_size_b = 8  # default
    if "70b" in model_family:
        model_size_b = 70
    elif "13b" in model_family:
        model_size_b = 13
    elif "8b" in model_family:
        model_size_b = 8
    elif "7b" in model_family:
        model_size_b = 7
    elif "405b" in model_family:
        model_size_b = 405
    
    # Strategy based on observations:
    # - TP should max out intra-node (NVLink) for fast all-reduce
    # - Minimize inter-node communication by keeping dp low
    # - PP=1 unless model is too large to fit
    # - eager mode has been better than inductor so far
    
    # Bandwidth ratio determines how much we should prefer intra-node parallelism
    bw_ratio = intra_bw / max(inter_bw, 1)
    
    # Set TP to fill the node (all intra-node communication via NVLink)
    tp = gpus_per_node
    tp = pick_valid("tp", tp)
    
    # Default: no pipeline parallelism
    pp = 1
    pp = pick_valid("pp", pp)
    
    # For very large models, may need PP
    if model_size_b >= 70:
        # Need more parallelism for memory
        if gpu_memory_gb <= 40:
            pp = pick_valid("pp", 4)
        else:
            pp = pick_valid("pp", 2)
    
    # dp uses remaining GPUs
    dp_target = total_gpus // (tp * pp)
    dp = pick_valid("dp", max(1, dp_target))
    
    # Ensure product = total_gpus; if not, adjust
    product = tp * dp * pp
    if product != total_gpus:
        # Try reducing tp
        for tp_try in sorted(valid.get("tp", [gpus_per_node]), reverse=True):
            for pp_try in sorted(valid.get("pp", [1])):
                dp_try = total_gpus // (tp_try * pp_try)
                if dp_try in valid.get("dp", [dp_try]) and tp_try * dp_try * pp_try == total_gpus:
                    if tp_try <= gpus_per_node:  # keep TP intra-node
                        tp, dp, pp = tp_try, dp_try, pp_try
                        break
            else:
                continue
            break
    
    # Micro batch size
    if pp <= 1:
        micro_batch = pick_valid("micro_batch", 1)
    else:
        per_gpu_batch = batch_size // dp
        # More micro batches reduces bubble ratio: aim for ~4*pp micro batches
        target_micro = max(1, per_gpu_batch // (4 * pp))
        micro_batch = pick_valid("micro_batch", target_micro)
    
    # Compile mode: try inductor with the optimal tp=4 config this time
    # Previous best was eager with tp=4, dp=4. Let's test inductor with same parallelism.
    compile_mode = pick_valid("compile_mode", "inductor")
    
    # Activation checkpointing: only if memory is tight
    bytes_per_param = 2  # bf16
    model_mem_gb = (model_size_b * 1e9 * bytes_per_param) / (tp * 1e9)
    optimizer_mem_gb = (model_size_b * 1e9 * bytes_per_param * 3) / (dp * tp * 1e9)  # Adam: 3x model
    per_gpu_batch = batch_size // (dp * pp)
    # Rough activation memory estimate
    activation_mem_gb = per_gpu_batch * seq_len * 4096 * 2 / (tp * 1e9) * workload.get("num_layers", 32) * 0.1
    
    total_est_mem = model_mem_gb + optimizer_mem_gb + activation_mem_gb
    
    if total_est_mem > gpu_memory_gb * 0.85:
        activation_checkpointing = True
    else:
        activation_checkpointing = False
    
    activation_checkpointing = pick_valid("activation_checkpointing", activation_checkpointing)
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": compile_mode,
        "activation_checkpointing": activation_checkpointing,
    }
