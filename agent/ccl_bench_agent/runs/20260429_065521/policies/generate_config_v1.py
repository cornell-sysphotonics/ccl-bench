
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    gpu_memory_gb = environment.get("gpu_memory_gb", 80)
    batch_size = workload.get("batch_size", 1)
    seq_len = workload.get("seq_len", 1024)
    model_family = workload.get("model_family", "").lower()
    
    # Build a lookup of valid choices for each config key
    config_space = workload.get("config_space", [])
    valid = {}
    for dim in config_space:
        valid[dim["key"]] = dim["choices"]
    
    # Helper to pick closest valid choice
    def pick_valid(key, target):
        choices = valid.get(key, [target])
        # For numeric types, pick the closest
        if all(isinstance(c, (int, float)) for c in choices):
            return min(choices, key=lambda c: abs(c - target))
        # For string/bool, return target if valid, else first choice
        if target in choices:
            return target
        return choices[0]
    
    # Estimate model size in billions of parameters
    model_size_b = 8  # default for Llama-3.1-8B
    if "70b" in model_family:
        model_size_b = 70
    elif "13b" in model_family:
        model_size_b = 13
    elif "8b" in model_family:
        model_size_b = 8
    elif "7b" in model_family:
        model_size_b = 7
    
    # Memory estimation per GPU (rough): model params in bf16 = 2 bytes/param
    # With FSDP (dp sharding), model memory is divided by dp
    # With TP, model memory is divided by tp
    # Activations scale with batch_size_per_gpu * seq_len / tp
    
    # Strategy: minimize communication overhead
    # - TP should stay within a node (NVLink is fast)
    # - Prefer smaller TP if memory allows (less all-reduce overhead)
    # - Use FSDP (dp) for the rest
    # - PP=1 to avoid pipeline bubbles unless model is very large
    
    # For 8B model on A100 40GB:
    # bf16 model weights: ~16GB total
    # With tp=2: ~8GB per GPU for weights, leaves room for activations
    # With FSDP dp=8: further sharding of optimizer states
    
    pp = 1
    
    # Determine TP based on model size and GPU memory
    if model_size_b >= 70:
        tp = min(gpus_per_node, 8)
    elif model_size_b >= 13:
        tp = min(gpus_per_node, 4)
    elif gpu_memory_gb <= 40:
        # Tight memory - use tp=2 for 8B models on 40GB GPUs
        tp = 2
    else:
        tp = 2
    
    tp = pick_valid("tp", tp)
    pp = pick_valid("pp", pp)
    
    # dp uses remaining GPUs
    dp = total_gpus // (tp * pp)
    dp = pick_valid("dp", dp)
    
    # Ensure tp * dp * pp = total_gpus, adjust if needed
    product = tp * dp * pp
    if product != total_gpus:
        # Try to fix dp
        dp = total_gpus // (tp * pp)
        dp = pick_valid("dp", dp)
    
    # Micro batch size (only matters if pp > 1, but set it anyway)
    # For pp=1, set to something reasonable
    if pp == 1:
        micro_batch = pick_valid("micro_batch", 1)
    else:
        # More micro batches = less bubble, but more memory
        num_micro = max(4, pp * 2)
        per_gpu_batch = batch_size // dp
        micro_batch = max(1, per_gpu_batch // num_micro)
        micro_batch = pick_valid("micro_batch", micro_batch)
    
    # Use inductor for better kernel fusion
    compile_mode = pick_valid("compile_mode", "inductor")
    
    # Activation checkpointing: enable for tight memory situations
    # 8B model, tp=2, A100 40GB, batch=32, dp=8 -> 4 samples/gpu
    # This might be tight, enable checkpointing to be safe and potentially
    # allow the run to succeed with good throughput
    bytes_per_param = 2  # bf16
    model_mem_gb = (model_size_b * 1e9 * bytes_per_param) / (tp * 1e9)
    optimizer_mem_gb = model_mem_gb * 3  # Adam states (divided by dp via FSDP)
    optimizer_mem_gb /= dp
    per_gpu_batch = batch_size // (dp * pp)
    activation_mem_gb = per_gpu_batch * seq_len * 0.001  # rough estimate
    
    total_est_mem = model_mem_gb + optimizer_mem_gb + activation_mem_gb
    
    if total_est_mem > gpu_memory_gb * 0.8:
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
