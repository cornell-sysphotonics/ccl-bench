
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
    if "405b" in model_family:
        model_size_b = 405
    elif "70b" in model_family:
        model_size_b = 70
    elif "13b" in model_family:
        model_size_b = 13
    elif "8b" in model_family:
        model_size_b = 8
    elif "7b" in model_family:
        model_size_b = 7
    
    # Bandwidth ratio determines how much we should avoid inter-node communication
    bw_ratio = intra_bw / max(inter_bw, 1)  # e.g., 2400/200 = 12x
    
    num_nodes = total_gpus // gpus_per_node  # 4 nodes
    
    # ---- Strategy for this iteration ----
    # History results (Llama-3.1-8B, 16 GPUs, 4 per node):
    #   tp=4, dp=4, pp=1, eager, AC=False → 8.793 (BEST)
    #   tp=4, dp=4, pp=1, inductor, AC=False → 9.009
    #   tp=4, dp=4, pp=1, eager, AC=True → 8.992
    #   tp=2, dp=8, pp=1, inductor, AC=False → 10.6
    #
    # Try: tp=4, dp=2, pp=2 to reduce inter-node allreduce at cost of pipeline bubble
    # With batch=32, dp=2: per-rank batch=16, micro_batch=2 → 8 microbatches (good for pp=2)
    
    # For general logic:
    # 1. TP fills intra-node (gpus_per_node) for NVLink
    # 2. Try to minimize inter-node dp by using some pp
    # 3. eager > inductor, AC=False is better for 8B
    
    tp = min(gpus_per_node, total_gpus)
    tp = pick_valid("tp", tp)
    
    # For small/medium models (< 30B), try pp=2 to cut dp in half when we have many nodes
    # For very large models, use more pp
    if model_size_b >= 70:
        if gpu_memory_gb <= 40:
            pp = pick_valid("pp", 4)
        else:
            pp = pick_valid("pp", 2)
    elif model_size_b >= 8 and num_nodes >= 4 and bw_ratio >= 10:
        # Inter-node is relatively slow; using pp=2 cuts dp communication in half
        pp = pick_valid("pp", 2)
    else:
        pp = 1
    
    pp = pick_valid("pp", pp)
    
    # dp uses remaining GPUs
    dp_target = total_gpus // (tp * pp)
    dp = pick_valid("dp", max(1, dp_target))
    
    # Verify product equals total_gpus; if not, search for best combo
    product = tp * dp * pp
    if product != total_gpus:
        best_combo = None
        best_score = float('inf')
        for tp_try in sorted(valid.get("tp", [gpus_per_node]), reverse=True):
            if tp_try > gpus_per_node:
                continue
            for pp_try in sorted(valid.get("pp", [1])):
                dp_try = total_gpus // (tp_try * pp_try)
                if dp_try in valid.get("dp", [dp_try]) and tp_try * dp_try * pp_try == total_gpus:
                    score = dp_try * (inter_bw / intra_bw) + pp_try * 0.5
                    if score < best_score:
                        best_score = score
                        best_combo = (tp_try, dp_try, pp_try)
        if best_combo:
            tp, dp, pp = best_combo
    
    # Micro batch size - optimize for pipeline efficiency
    if pp <= 1:
        micro_batch = pick_valid("micro_batch", 1)
    else:
        per_gpu_batch = batch_size // dp
        # Want many micro-batches relative to pp to minimize bubble
        # Ideal: num_microbatches >= 4*pp
        # per_gpu_batch / micro_batch = num_microbatches
        # So micro_batch = per_gpu_batch / (4*pp)
        target_micro = max(1, per_gpu_batch // (4 * pp))
        micro_batch = pick_valid("micro_batch", target_micro)
        # Make sure micro_batch doesn't exceed per_gpu_batch
        if micro_batch > per_gpu_batch:
            micro_batch = pick_valid("micro_batch", per_gpu_batch)
    
    # Compile mode: eager has been consistently better
    compile_mode = pick_valid("compile_mode", "eager")
    
    # Activation checkpointing: False is better for 8B (enough memory)
    # Only enable for very large models or tight memory
    if model_size_b >= 30 and gpu_memory_gb <= 40:
        activation_checkpointing = pick_valid("activation_checkpointing", True)
    else:
        activation_checkpointing = pick_valid("activation_checkpointing", False)
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": compile_mode,
        "activation_checkpointing": activation_checkpointing,
    }
