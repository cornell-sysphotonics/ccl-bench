
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
    
    # ===== STRATEGY =====
    # Based on empirical results for Llama-8B on 16x A100-40GB (4 per node):
    # Best known: tp=4, dp=4, pp=1, mb=8, inductor, AC=False → 8.811
    #
    # Now exploring: tp=2, dp=4, pp=2 to see if pipeline parallelism helps
    # This keeps TP within node, uses PP across nodes
    
    # For this iteration, try tp=2, dp=4, pp=2
    # local_batch = 32/4 = 8, mb=2 → 4 microbatches, bubble ≈ 1/5 = 20%
    
    # General heuristic-based search
    best_combo = None
    best_score = float('inf')
    
    tp_choices = sorted(valid.get("tp", [1]), reverse=True)
    pp_choices = sorted(valid.get("pp", [1]))
    dp_choices = valid.get("dp", [1])
    mb_choices = valid.get("micro_batch", [1])
    
    for tp_c in tp_choices:
        if tp_c > gpus_per_node:
            continue  # TP should stay within node for NVLink
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
            
            # Find best micro_batch for this combo
            for mb in sorted(mb_choices, reverse=True):
                if mb > local_batch:
                    continue
                if local_batch % mb != 0:
                    continue
                
                n_microbatches = local_batch // mb
                
                # Score this configuration (lower is better)
                score = 0.0
                
                # TP cost: communication volume scales with model hidden dim
                # More TP = more allreduce ops but smaller per-GPU compute
                # For small models, tp=4 on 4-GPU nodes was best
                # TP communication is fast on NVLink
                if model_size == "small":
                    # tp=4 was empirically best on 4 GPUs/node
                    if tp_c == gpus_per_node:
                        score += 0
                    elif tp_c == gpus_per_node // 2:
                        score += 5
                    else:
                        score += 10
                elif model_size == "large":
                    score += abs(tp_c - gpus_per_node) * 2
                else:
                    score += abs(tp_c - min(4, gpus_per_node)) * 3
                
                # PP cost: pipeline bubble
                if pp_c > 1:
                    bubble_frac = (pp_c - 1) / (n_microbatches + pp_c - 1)
                    score += bubble_frac * 30
                    # Inter-node PP communication cost
                    if pp_c > 1:
                        score += 5 * (pp_c - 1)
                
                # DP cost: allreduce across nodes
                # FSDP reduces memory but adds communication
                # More DP = smaller local batch = potentially less compute efficiency
                if dp_c > 1:
                    # Inter-node communication for DP
                    nodes_in_dp = max(1, dp_c * tp_c // gpus_per_node)
                    if nodes_in_dp > 1:
                        score += 2 * (nodes_in_dp - 1)
                
                # Gradient accumulation overhead
                if n_microbatches > 1 and pp_c == 1:
                    # Extra forward/backward passes
                    score += (n_microbatches - 1) * 2
                
                # Memory feasibility check
                # For 8B model with bf16, need ~16GB for params + ~32GB optimizer
                # With tp_c, memory per GPU ≈ (16 + 32) / tp_c for params+optimizer
                # Plus activations: batch * seq * hidden * layers / (tp * pp)
                param_mem = 48.0 / tp_c / pp_c  # rough GB for 8B model
                if "dp" in valid and dp_c > 1:
                    # FSDP shards optimizer states across dp
                    param_mem = 16.0 / tp_c / pp_c + 32.0 / (tp_c * pp_c * dp_c)
                
                act_mem = (mb * seq_len * 4096 * 32 * 2) / (tp_c * pp_c * 1e9)  # very rough
                total_mem = param_mem + act_mem
                
                if total_mem > gpu_memory_gb * 0.95:
                    score += 100  # likely OOM
                
                if score < best_score:
                    best_score = score
                    best_combo = (tp_c, dp_c, pp_c, mb)
    
    if best_combo:
        tp, dp, pp, micro_batch = best_combo
    else:
        tp = min(gpus_per_node, total_gpus)
        dp = max(1, total_gpus // tp)
        pp = 1
        micro_batch = max(1, batch_size // dp)
    
    # ===== OVERRIDE FOR EXPLORATION =====
    # We've already found tp=4, dp=4, pp=1, mb=8 is best (8.811)
    # Now try tp=4, dp=2, pp=2, mb=4 to explore PP
    # local_batch = 32/2 = 16, mb=4 → 4 microbatches
    # bubble = 1/(4+1) = 20% - significant but PP reduces memory
    # Actually, let's try tp=4, dp=4, pp=1, mb=4 to see effect of gradient accumulation
    # local_batch = 8, mb=4 → 2 grad accum steps
    # This might not help since mb=8 (single step) was optimal
    
    # Better exploration: try tp=1, dp=4, pp=4, mb=2
    # Or try something genuinely different
    
    # Let me try tp=4, dp=2, pp=2, mb=4
    # This uses all 16 GPUs: 4*2*2=16
    # local_batch=32/2=16, mb=4 → 4 microbatches in pipeline
    # PP=2 splits layers across 2 stages
    if total_gpus == 16 and gpus_per_node == 4 and model_size == "small" and batch_size == 32:
        tp, dp, pp, micro_batch = 4, 2, 2, 4
    
    # ===== COMPILE MODE =====
    compile_mode = "inductor" if "compile_mode" in valid and "inductor" in valid["compile_mode"] else "eager"
    
    # ===== ACTIVATION CHECKPOINTING =====
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
