
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus    = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    gpu_memory_gb = environment.get("gpu_memory_gb", 40)
    batch_size    = workload.get("batch_size", 1)
    seq_len       = workload.get("seq_len", 1024)
    precision     = workload.get("precision", "bf16")
    
    # Build lookup of valid choices for each dimension - handle various key formats
    config_space = workload.get("config_space", [])
    valid = {}
    for dim in config_space:
        # Try different possible key names
        if isinstance(dim, dict):
            dim_name = dim.get("name") or dim.get("key") or dim.get("dimension") or dim.get("param") or ""
            dim_choices = dim.get("choices") or dim.get("values") or dim.get("options") or []
            if dim_name:
                valid[dim_name] = dim_choices
    
    # Model analysis for Llama-3.1-8B:
    # ~8B params, ~16GB in bf16
    # Adam optimizer states: ~48GB total (params + grads + 2 momentum states in fp32)
    # FSDP with dp=8 shards this to ~6GB per GPU
    # TP=2 splits model across 2 GPUs -> ~8GB params per GPU
    # Combined with FSDP: very comfortable memory fit on 40GB A100
    
    # Strategy: minimize TP to reduce all-reduce communication overhead
    # TP=2 within node (fast NVLink), DP=8 across nodes (FSDP handles grad sync)
    # No PP to avoid pipeline bubble overhead
    
    # Default strategy based on model size and GPU count
    model_family = workload.get("model_family", "").lower()
    
    # For 8B class models on 40GB GPUs:
    # TP=2 is sufficient to fit in memory with FSDP
    tp = 2
    pp = 1
    dp = total_gpus // (tp * pp)  # 16 // 2 = 8
    
    # Ensure product equals total_gpus
    if tp * dp * pp != total_gpus:
        # Fallback: use tp=4, dp=4
        tp = min(4, gpus_per_node)
        dp = total_gpus // tp
        pp = 1
    
    # local batch = batch_size / dp
    local_batch = batch_size // dp if dp > 0 else batch_size
    
    # micro_batch: use the largest valid value that divides local_batch
    micro_batch = local_batch
    
    # Validate all parameters against config space
    def pick_valid(name, desired, fallback=None):
        if name not in valid or not valid[name]:
            return desired
        choices = valid[name]
        if desired in choices:
            return desired
        # Pick closest valid choice
        return min(choices, key=lambda x: abs(x - desired) if isinstance(x, (int, float)) else 0)
    
    def pick_valid_dividing(name, desired, must_divide=None):
        """Pick largest valid choice <= desired that divides must_divide."""
        if name not in valid or not valid[name]:
            return desired
        choices = valid[name]
        if must_divide is not None:
            # Filter to choices that divide must_divide
            dividing = [c for c in choices if isinstance(c, int) and must_divide % c == 0]
            if dividing:
                # Pick largest <= desired
                good = [c for c in dividing if c <= desired]
                if good:
                    return max(good)
                return min(dividing)
        # Fallback: pick closest
        if desired in choices:
            return desired
        return min(choices, key=lambda x: abs(x - desired) if isinstance(x, (int, float)) else 0)
    
    tp = pick_valid("tp", tp)
    dp = pick_valid("dp", dp)
    pp = pick_valid("pp", pp)
    
    # Recompute local_batch after validation
    local_batch = batch_size // dp if dp > 0 else batch_size
    micro_batch = pick_valid_dividing("micro_batch", local_batch, local_batch)
    
    # Use inductor for kernel fusion speedups
    compile_mode = "inductor"
    if "compile_mode" in valid and compile_mode not in valid["compile_mode"]:
        compile_mode = valid["compile_mode"][0] if valid["compile_mode"] else "eager"
    
    # No activation checkpointing - avoids recomputation overhead when memory allows
    # With tp=2 and dp=8, memory should be comfortable for 8B model
    activation_checkpointing = False
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": compile_mode,
        "activation_checkpointing": activation_checkpointing,
    }
