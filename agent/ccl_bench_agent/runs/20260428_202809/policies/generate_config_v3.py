
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
        return choices[-1]
    
    # ===== PARALLELISM STRATEGY =====
    # Based on experiments:
    # - tp=4, dp=4, inductor gave best result (8.811) for Llama-8B on 16x A100-40GB
    # - tp=2, dp=8, inductor was much worse (10.27)
    # - tp=4, dp=4, eager was middle (9.111)
    # 
    # Strategy: Keep TP within node (NVLink), maximize TP for small models on 40GB,
    # use inductor for compilation speedup
    
    num_nodes = total_gpus // gpus_per_node if gpus_per_node > 0 else 1
    
    if model_size == "large":
        # Large models need full node TP + possibly PP across nodes
        tp = gpus_per_node
        pp = max(1, num_nodes // 2)
    elif model_size == "medium":
        tp = min(gpus_per_node, 4)
        pp = 1
    else:
        # Small models (8B): tp=4 within node was best on 40GB A100s
        if gpu_memory_gb >= 80:
            tp = min(gpus_per_node, 2)
        elif gpu_memory_gb >= 40:
            tp = min(gpus_per_node, 4)
        else:
            tp = gpus_per_node
        pp = 1
    
    tp = min(tp, total_gpus)
    tp = pick_valid("tp", tp)
    pp = pick_valid("pp", pp)
    
    # Compute dp
    dp = total_gpus // (tp * pp)
    
    # Validate the combination
    if dp < 1 or tp * dp * pp != total_gpus:
        # Search for valid combinations
        best_combo = None
        best_score = float('inf')
        
        for tp_c in sorted(valid.get("tp", [tp]), reverse=True):
            if tp_c > gpus_per_node:
                continue  # Keep TP within node
            for pp_c in sorted(valid.get("pp", [pp])):
                if tp_c * pp_c > total_gpus:
                    continue
                dp_c = total_gpus // (tp_c * pp_c)
                if dp_c < 1 or tp_c * dp_c * pp_c != total_gpus:
                    continue
                if "dp" in valid and dp_c not in valid["dp"]:
                    continue
                
                # Scoring heuristic: prefer tp=4 for small models, pp=1
                target_tp = 4 if model_size == "small" and gpu_memory_gb < 80 else (gpus_per_node if model_size == "large" else 2)
                score = abs(tp_c - target_tp) * 10 + pp_c * 5
                if score < best_score:
                    best_score = score
                    best_combo = (tp_c, dp_c, pp_c)
        
        if best_combo:
            tp, dp, pp = best_combo
    
    if "dp" in valid and dp not in valid["dp"]:
        dp = pick_valid("dp", dp)
    
    # ===== MICRO BATCH =====
    local_batch = batch_size // dp if dp > 0 else batch_size
    
    if "micro_batch" in valid:
        mb_choices = sorted(valid["micro_batch"], reverse=True)
        # Pick largest micro_batch that divides local_batch (minimize grad accumulation overhead)
        micro_batch = 1
        for mb in mb_choices:
            if mb <= local_batch and local_batch % mb == 0:
                micro_batch = mb
                break
        if micro_batch == 1 and local_batch > 1:
            for mb in mb_choices:
                if mb <= local_batch:
                    micro_batch = mb
                    break
    else:
        micro_batch = max(1, local_batch)
    
    # ===== COMPILE MODE =====
    # inductor was better than eager with tp=4, dp=4 (8.811 vs 9.111)
    compile_mode = "inductor" if "compile_mode" in valid and "inductor" in valid["compile_mode"] else "eager"
    
    # ===== ACTIVATION CHECKPOINTING =====
    # Try enabling AC to see if it helps - trades compute for memory
    # For this iteration, try AC=True to test if freeing memory helps throughput
    activation_checkpointing = True
    if "activation_checkpointing" in valid:
        if model_size == "small" and gpu_memory_gb >= 80:
            activation_checkpointing = False  # plenty of memory
        # For 40GB with 8B model, AC might help by reducing memory pressure
    
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
