
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
    
    num_nodes = total_gpus // gpus_per_node
    bw_ratio = intra_bw / max(inter_bw, 1)
    
    # ---- Strategy ----
    # Based on execution history for Llama-3.1-8B on 16 A100-40GB (4 nodes × 4 GPUs):
    #   tp=4, dp=4, pp=1, eager, AC=False → 8.793 (BEST)
    #   tp=4, dp=4, pp=1, inductor, AC=False → 9.009
    #   tp=4, dp=4, pp=1, eager, AC=True → 8.992
    #   tp=2, dp=8, pp=1, inductor, AC=False → 10.6
    #   tp=4, dp=2, pp=2 → FAILED
    #
    # This iteration: try tp=1, dp=16, pp=1 (pure FSDP) to check if less TP overhead helps
    # FSDP shards model across all 16 GPUs, reducing memory per GPU significantly
    # For 8B model: 16GB bf16 weights / 16 = 1GB per GPU - very memory efficient
    # Trade-off: allgather for each layer during forward/backward vs TP all-reduce
    
    # General strategy for different scenarios:
    # For small models (< 15B) with enough memory: prefer tp within node, dp across nodes, pp=1
    # For large models (>= 30B): may need pp and/or activation checkpointing
    
    if model_size_b <= 15:
        # Small model: TP within node is good, but let's also test pure FSDP
        # From history: tp=4,dp=4 (8.793) >> tp=2,dp=8 (10.6)
        # So tp=4 within node is beneficial. Let's keep our best config.
        # Try: tp=2, dp=8, pp=1, eager, AC=False - to see if eager helps tp=2
        tp = pick_valid("tp", 2)
        pp = pick_valid("pp", 1)
        dp_target = total_gpus // (tp * pp)
        dp = pick_valid("dp", max(1, dp_target))
    elif model_size_b <= 30:
        tp = pick_valid("tp", gpus_per_node)
        pp = pick_valid("pp", 1)
        dp_target = total_gpus // (tp * pp)
        dp = pick_valid("dp", max(1, dp_target))
    elif model_size_b <= 70:
        tp = pick_valid("tp", gpus_per_node)
        pp = pick_valid("pp", 2)
        dp_target = total_gpus // (tp * pp)
        dp = pick_valid("dp", max(1, dp_target))
    else:
        tp = pick_valid("tp", gpus_per_node)
        pp = pick_valid("pp", 4)
        dp_target = total_gpus // (tp * pp)
        dp = pick_valid("dp", max(1, dp_target))
    
    # Verify product equals total_gpus; if not, find valid combo
    product = tp * dp * pp
    if product != total_gpus:
        best_combo = None
        best_score = float('inf')
        for tp_c in valid.get("tp", [4]):
            for pp_c in valid.get("pp", [1]):
                for dp_c in valid.get("dp", [1]):
                    if tp_c * dp_c * pp_c == total_gpus:
                        # Prefer tp within node, minimize dp (inter-node), minimize pp
                        tp_penalty = 0 if tp_c <= gpus_per_node else 100
                        score = tp_penalty + dp_c * 2 + pp_c * 3
                        if score < best_score:
                            best_score = score
                            best_combo = (tp_c, dp_c, pp_c)
        if best_combo:
            tp, dp, pp = best_combo
    
    # Micro batch size
    if pp <= 1:
        micro_batch = pick_valid("micro_batch", 1)
    else:
        per_gpu_batch = batch_size // dp
        target_micro = max(1, per_gpu_batch // (4 * pp))
        micro_batch = pick_valid("micro_batch", target_micro)
        if micro_batch > per_gpu_batch:
            micro_batch = pick_valid("micro_batch", per_gpu_batch)
    
    # Compile mode: eager consistently better
    compile_mode = pick_valid("compile_mode", "eager")
    
    # Activation checkpointing: only for large models or tight memory
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
