
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    gpu_memory_gb = environment.get("gpu_memory_gb", 80)
    batch_size = workload.get("batch_size", 1)
    seq_len = workload.get("seq_len", 2048)
    precision = workload.get("precision", "bf16")
    intra_bw = environment.get("intra_node_bandwidth_gbps", 300)
    inter_bw = environment.get("inter_node_bandwidth_gbps", 100)
    
    # Build lookup of valid choices per key
    config_space = workload.get("config_space", [])
    valid = {}
    for dim in config_space:
        valid[dim["key"]] = dim["choices"]
    
    model_family = workload.get("model_family", "").lower()
    
    # Determine model size category
    if "70b" in model_family or "65b" in model_family:
        model_size = "large"
    elif "13b" in model_family or "30b" in model_family or "34b" in model_family:
        model_size = "medium"
    else:
        model_size = "small"  # 7B, 8B class
    
    # Helper to pick closest valid choice
    def pick_valid(key, target):
        if key not in valid:
            return target
        choices = sorted(valid[key])
        # Pick closest
        best = choices[0]
        for c in choices:
            if abs(c - target) < abs(best - target):
                best = c
        return best
    
    def pick_valid_leq(key, target):
        """Pick largest valid choice <= target"""
        if key not in valid:
            return target
        choices = sorted(valid[key], reverse=True)
        for c in choices:
            if c <= target:
                return c
        return choices[-1]  # smallest available
    
    # Strategy: maximize intra-node communication (TP within node), 
    # use DP across nodes, minimize PP unless memory requires it
    
    # TP should stay within a node for NVLink benefit
    # For small models on 40GB GPUs, tp=4 (full node) worked best
    # This maximizes intra-node utilization and reduces inter-node DP traffic
    
    if model_size == "large":
        # Large models need high TP for memory
        tp = min(gpus_per_node, 8)
        pp = max(1, total_gpus // (gpus_per_node * 2))  # some PP across nodes
    elif model_size == "medium":
        tp = min(gpus_per_node, 4)
        pp = 1
    else:
        # Small models (8B): tp=4 on 40GB worked well
        # Key insight: tp=4 within node, dp=4 across nodes beat tp=2, dp=8
        if gpu_memory_gb >= 80:
            tp = min(gpus_per_node, 2)  # less TP needed on bigger GPUs
        elif gpu_memory_gb >= 40:
            tp = min(gpus_per_node, 4)  # tp=4 within node
        else:
            tp = gpus_per_node
        pp = 1
    
    # Clamp tp to valid and to gpus_per_node
    tp = min(tp, gpus_per_node, total_gpus)
    tp = pick_valid("tp", tp)
    
    # Compute dp from remaining GPUs
    remaining = total_gpus // tp
    dp = remaining // pp if pp > 0 else remaining
    
    # Validate dp
    if "dp" in valid:
        dp_choices = sorted(valid["dp"])
        if dp not in dp_choices:
            dp = pick_valid_leq("dp", dp)
    
    # Validate pp
    pp = pick_valid("pp", pp)
    
    # Ensure tp * dp * pp = total_gpus, try to fix if not
    if tp * dp * pp != total_gpus:
        best_combo = None
        best_waste = total_gpus + 1
        for tp_c in sorted(valid.get("tp", [tp]), reverse=True):
            for pp_c in sorted(valid.get("pp", [pp])):
                dp_c = total_gpus // (tp_c * pp_c) if tp_c * pp_c <= total_gpus else 0
                if dp_c >= 1 and tp_c * dp_c * pp_c == total_gpus:
                    if "dp" not in valid or dp_c in valid["dp"]:
                        # Prefer tp within node
                        if tp_c <= gpus_per_node:
                            waste = 0
                            if best_combo is None or waste < best_waste:
                                best_combo = (tp_c, dp_c, pp_c)
                                best_waste = waste
        if best_combo:
            tp, dp, pp = best_combo
            # Among valid combos, prefer our original tp choice
            # Re-search with preference
            candidates = []
            for tp_c in sorted(valid.get("tp", [tp]), reverse=True):
                for pp_c in sorted(valid.get("pp", [pp])):
                    dp_c = total_gpus // (tp_c * pp_c) if tp_c * pp_c <= total_gpus else 0
                    if dp_c >= 1 and tp_c * dp_c * pp_c == total_gpus:
                        if "dp" not in valid or dp_c in valid["dp"]:
                            if tp_c <= gpus_per_node:
                                candidates.append((tp_c, dp_c, pp_c))
            # Prefer: tp=4 for small models on 40GB (proven best)
            if candidates:
                # Sort by preference: tp close to target, pp=1 preferred
                target_tp = 4 if model_size == "small" and gpu_memory_gb < 80 else 2
                candidates.sort(key=lambda x: (x[2], abs(x[0] - target_tp)))
                tp, dp, pp = candidates[0]
    
    # micro_batch: must divide local_batch_size = batch_size / dp
    local_batch = batch_size // dp if dp > 0 else batch_size
    
    if "micro_batch" in valid:
        mb_choices = sorted(valid["micro_batch"], reverse=True)
        # Pick largest micro_batch that divides local_batch for efficiency
        micro_batch = 1
        for mb in mb_choices:
            if mb <= local_batch and local_batch % mb == 0:
                micro_batch = mb
                break
        # If none divides cleanly, pick the largest <= local_batch
        if micro_batch == 1 and local_batch > 1:
            for mb in mb_choices:
                if mb <= local_batch:
                    micro_batch = mb
                    break
    else:
        micro_batch = max(1, local_batch)
    
    # compile_mode: try inductor with the best parallelism config
    # Previous run showed inductor was worse with tp=2/dp=8, but let's test with tp=4/dp=4
    compile_mode = "inductor" if "compile_mode" in valid and "inductor" in valid["compile_mode"] else "eager"
    
    # Activation checkpointing: disable for speed unless memory-constrained
    activation_checkpointing = False
    if "activation_checkpointing" in valid:
        # Enable for large models or very tight memory
        if model_size == "large" or (model_size == "medium" and gpu_memory_gb < 40):
            activation_checkpointing = True
    
    config = {}
    if "tp" in valid:
        config["tp"] = tp
    if "dp" in valid:
        config["dp"] = dp
    if "pp" in valid:
        config["pp"] = pp
    if "micro_batch" in valid:
        config["micro_batch"] = micro_batch
    if "compile_mode" in valid:
        config["compile_mode"] = compile_mode
    if "activation_checkpointing" in valid:
        config["activation_checkpointing"] = activation_checkpointing
    
    return config
