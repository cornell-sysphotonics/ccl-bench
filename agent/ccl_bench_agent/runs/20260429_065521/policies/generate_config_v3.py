
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
    
    # Lessons learned from history on this workload:
    # - tp=4, dp=4, pp=1, eager, no AC: 8.793 (best)
    # - tp=2, dp=8, pp=1, inductor, no AC: 10.6 (worse - more inter-node comm)
    # - tp=4, dp=4, pp=1, inductor, no AC: 9.009 (inductor slightly worse than eager)
    #
    # General strategy:
    # 1. TP should fill intra-node (max gpus_per_node) for NVLink benefit
    # 2. Minimize dp to reduce inter-node allreduce, but need enough for memory
    # 3. PP=1 to avoid bubble overhead for smaller models
    # 4. eager > inductor for this workload
    # 5. Test activation_checkpointing=True: reduces memory, may allow faster compute
    
    # Bandwidth ratio
    bw_ratio = intra_bw / max(inter_bw, 1)
    
    # Set TP to fill the node
    tp = gpus_per_node
    tp = pick_valid("tp", tp)
    
    # Default: no pipeline parallelism for models that fit
    pp = 1
    
    # For very large models, use PP
    if model_size_b >= 70:
        if gpu_memory_gb <= 40:
            pp = pick_valid("pp", 4)
        else:
            pp = pick_valid("pp", 2)
    
    pp = pick_valid("pp", pp)
    
    # dp uses remaining GPUs
    dp_target = total_gpus // (tp * pp)
    dp = pick_valid("dp", max(1, dp_target))
    
    # Ensure product = total_gpus; if not, adjust
    product = tp * dp * pp
    if product != total_gpus:
        best_combo = None
        best_score = float('inf')
        for tp_try in sorted(valid.get("tp", [gpus_per_node]), reverse=True):
            if tp_try > gpus_per_node:
                continue  # Keep TP intra-node
            for pp_try in sorted(valid.get("pp", [1])):
                dp_try = total_gpus // (tp_try * pp_try)
                if dp_try in valid.get("dp", [dp_try]) and tp_try * dp_try * pp_try == total_gpus:
                    # Score: prefer high TP (intra-node), low PP (no bubble), moderate dp
                    # Penalize inter-node communication (dp) and pipeline bubbles (pp)
                    score = dp_try * (inter_bw / intra_bw) + pp_try * 0.5
                    if score < best_score:
                        best_score = score
                        best_combo = (tp_try, dp_try, pp_try)
        if best_combo:
            tp, dp, pp = best_combo
    
    # Micro batch size - only matters for PP > 1
    if pp <= 1:
        micro_batch = pick_valid("micro_batch", 1)
    else:
        per_gpu_batch = batch_size // dp
        # More micro batches reduces bubble ratio
        target_micro = max(1, per_gpu_batch // (4 * pp))
        micro_batch = pick_valid("micro_batch", target_micro)
    
    # Compile mode: eager has been better than inductor
    compile_mode = pick_valid("compile_mode", "eager")
    
    # Activation checkpointing: test True this iteration
    # For small-medium models on limited memory GPUs, AC can help by reducing
    # peak memory, potentially allowing better memory utilization
    # For 8B on 40GB A100 with tp=4: model ~4GB per GPU, should have room
    # Let's try AC=True to see if it helps or hurts
    activation_checkpointing = pick_valid("activation_checkpointing", True)
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": compile_mode,
        "activation_checkpointing": activation_checkpointing,
    }
