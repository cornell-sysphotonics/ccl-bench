
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
    
    num_nodes = total_gpus // gpus_per_node if gpus_per_node > 0 else 1
    
    # ===== SEARCH BEST VALID PARALLELISM COMBINATION =====
    # For this workload (Llama-8B, 16 GPUs, 4 per node, A100-40GB):
    # Known results:
    #   tp=4, dp=4, pp=1, mb=8, inductor, AC=False → 8.811 (best)
    #   tp=4, dp=4, pp=1, mb=8, eager, AC=False → 9.111
    #   tp=4, dp=4, pp=1, mb=8, inductor, AC=True → 9.048
    #   tp=2, dp=8, pp=1, mb=4, inductor, AC=False → 10.27
    #
    # Try: tp=4, dp=2, pp=2, micro_batch=4, inductor, AC=False
    # This tests whether PP=2 across nodes can help
    
    best_combo = None
    best_score = float('inf')
    
    tp_choices = sorted(valid.get("tp", [1]), reverse=True)
    pp_choices = sorted(valid.get("pp", [1]))
    dp_choices = valid.get("dp", list(range(1, total_gpus + 1)))
    
    for tp_c in tp_choices:
        for pp_c in pp_choices:
            if tp_c * pp_c > total_gpus:
                continue
            dp_c = total_gpus // (tp_c * pp_c)
            if dp_c < 1 or tp_c * dp_c * pp_c != total_gpus:
                continue
            if "dp" in valid and dp_c not in valid["dp"]:
                continue
            
            local_batch = batch_size // dp_c
            if local_batch < 1:
                continue
            
            # Score the combination
            # Penalize TP spanning across nodes (slow inter-node for TP)
            tp_cross_node = tp_c > gpus_per_node
            if tp_cross_node:
                score = 1000  # very bad
            else:
                score = 0
                
                # TP communication cost (within node is fast)
                # More TP = more communication but less memory per GPU
                # For small models, tp=gpus_per_node is ideal (fills node)
                if model_size == "small":
                    # tp=4 (full node) was best for 4 GPUs/node
                    score += abs(tp_c - gpus_per_node) * 10
                elif model_size == "medium":
                    score += abs(tp_c - min(gpus_per_node, 4)) * 10
                else:
                    score += abs(tp_c - gpus_per_node) * 5
                
                # PP adds pipeline bubble overhead
                # bubble_fraction ≈ (pp-1) / num_microbatches
                # With pp=2 and local_batch=16, mb=4: 4 microbatches, bubble=25%
                # With pp=1: no bubble
                if pp_c > 1:
                    # Estimate number of microbatches
                    max_mb = min(local_batch, max(valid.get("micro_batch", [1])))
                    n_microbatches = max(1, local_batch // max_mb)
                    bubble_frac = (pp_c - 1) / (n_microbatches + pp_c - 1)
                    score += bubble_frac * 50
                    # Also penalize inter-node PP communication
                    score += pp_c * 3
                
                # More DP = more all-reduce communication (across nodes)
                if dp_c > num_nodes:
                    score += dp_c * 0.5
            
            if score < best_score:
                best_score = score
                best_combo = (tp_c, dp_c, pp_c)
    
    if best_combo:
        tp, dp, pp = best_combo
    else:
        tp, dp, pp = gpus_per_node, total_gpus // gpus_per_node, 1
    
    # ===== MICRO BATCH =====
    local_batch = batch_size // dp if dp > 0 else batch_size
    
    if "micro_batch" in valid:
        mb_choices = sorted(valid["micro_batch"], reverse=True)
        micro_batch = 1
        # Pick largest micro_batch that divides local_batch
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
    
    # For PP > 1, prefer smaller micro_batch to fill pipeline better
    if pp > 1 and "micro_batch" in valid:
        # Want many microbatches to reduce bubble
        # Target at least 2*pp microbatches
        target_n_mb = 2 * pp
        target_mb = max(1, local_batch // target_n_mb)
        mb_choices_asc = sorted(valid["micro_batch"])
        for mb in mb_choices_asc:
            if mb >= target_mb and mb <= local_batch and local_batch % mb == 0:
                micro_batch = mb
                break
    
    # ===== COMPILE MODE =====
    # inductor consistently better than eager
    compile_mode = "inductor" if "compile_mode" in valid and "inductor" in valid["compile_mode"] else "eager"
    
    # ===== ACTIVATION CHECKPOINTING =====
    # AC=False is better for small models with enough memory (avoids recompute overhead)
    # Only enable for large models or tight memory
    if model_size == "large" or (model_size == "medium" and gpu_memory_gb < 40):
        activation_checkpointing = True
    else:
        activation_checkpointing = False
    
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
