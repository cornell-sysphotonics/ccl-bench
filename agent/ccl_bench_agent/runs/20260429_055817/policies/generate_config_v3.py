
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    gpu_memory_gb = environment.get("gpu_memory_gb", 80)
    batch_size = workload.get("batch_size", 1)
    seq_len = workload.get("seq_len", 1024)
    model_family = workload.get("model_family", "")
    precision = workload.get("precision", "fp32")
    num_layers = workload.get("num_layers", 32)
    num_heads = workload.get("num_heads", 32)

    # Build lookup of valid choices per key
    config_space = workload.get("config_space", [])
    valid = {}
    for dim in config_space:
        valid[dim["key"]] = dim["choices"]

    def pick_valid(key, target):
        """Pick closest valid choice for a key."""
        choices = valid.get(key, [target])
        return min(choices, key=lambda x: abs(x - target))

    # Estimate model size
    model_str = str(model_family).lower()
    if "70b" in model_str:
        param_billions = 70
    elif "13b" in model_str:
        param_billions = 13
    elif "8b" in model_str:
        param_billions = 8
    elif "7b" in model_str:
        param_billions = 7
    elif "3b" in model_str:
        param_billions = 3
    elif "1b" in model_str or "1.5b" in model_str:
        param_billions = 1.5
    else:
        param_billions = 8

    bytes_per_param = 2 if precision in ["bf16", "fp16"] else 4
    model_memory_gb = param_billions * bytes_per_param  # params in GB

    # For Llama-3.1-8B on 16x A100-40GB (4 per node):
    # History shows:
    #   tp=4, dp=4, pp=1, inductor, act_ckpt=True → 8.817
    #   tp=1, dp=16, pp=1, inductor, act_ckpt=False → 10.94
    # 
    # Strategy: Try tp=2, dp=8 to reduce TP communication overhead
    # while using FSDP across more GPUs. FSDP communication can overlap
    # with computation, while TP all-reduce is on the critical path.
    
    # Memory estimation for feasibility
    # Per-GPU with tp and dp (FSDP):
    # - Params: model_memory_gb / tp (TP shards params)
    # - Optimizer states: param_billions * 12 / (tp * dp) (FSDP shards these)
    # - Activations: depends on local batch, seq_len, hidden_dim, tp
    # - With activation checkpointing: activations reduced ~5x
    
    intra_bw = environment.get("intra_node_bandwidth_gbps", 600)
    inter_bw = environment.get("inter_node_bandwidth_gbps", 100)
    
    tp_choices = sorted(valid.get("tp", [1]))
    dp_choices = sorted(valid.get("dp", [1]), reverse=True)
    pp_choices = sorted(valid.get("pp", [1]))
    mb_choices = valid.get("micro_batch", [1])
    
    mem_limit = gpu_memory_gb * 0.85  # leave some headroom
    
    # Estimate hidden_dim from model
    if param_billions >= 65:
        hidden_dim = 8192
    elif param_billions >= 12:
        hidden_dim = 5120
    elif param_billions >= 7:
        hidden_dim = 4096
    elif param_billions >= 3:
        hidden_dim = 3200
    else:
        hidden_dim = 2048
    
    best_config = None
    best_cost = float('inf')
    
    for tp in tp_choices:
        if tp > gpus_per_node:
            continue  # TP should stay within node for NVLink
            
        for pp in pp_choices:
            dp_needed = total_gpus // (tp * pp)
            if dp_needed < 1 or dp_needed not in valid.get("dp", [1]):
                continue
            dp = dp_needed
            
            if tp * dp * pp != total_gpus:
                continue
            
            local_batch = batch_size // dp
            if local_batch < 1:
                continue
            
            # Memory check
            param_mem = model_memory_gb / tp
            opt_mem = (param_billions * 12) / (tp * dp)
            
            # Activation memory estimate (rough, in GB)
            act_mem = (local_batch * seq_len * hidden_dim * num_layers * 2) / (tp * 1e9)
            act_mem_ckpt = act_mem / 5.0
            
            # Try without activation checkpointing first
            for act_ckpt in [False, True]:
                eff_act_mem = act_mem_ckpt if act_ckpt else act_mem
                total_mem = param_mem + opt_mem + eff_act_mem
                
                if total_mem > mem_limit:
                    continue
                
                # Cost heuristic: estimate step time
                # TP communication: 2 * model_memory_gb/tp * num_layers * 2 / intra_bw per step
                # (all-reduce per layer, forward + backward)
                # DP communication: model_memory_gb / dp (FSDP all-gather + reduce-scatter)
                # PP communication: pipeline bubble overhead
                
                # TP cost: proportional to tp (more GPUs = more all-reduce)
                # Higher TP = less compute per GPU but more comm
                tp_comm_cost = 0
                if tp > 1:
                    # All-reduce volume per layer = 2 * hidden_dim * seq_len * local_batch * bytes_per_param
                    # For forward + backward = 2x
                    # Total = 2 * num_layers * 2 * hidden_dim * seq_len * local_batch * 2 / intra_bw
                    tp_comm_cost = (4 * num_layers * hidden_dim * seq_len * local_batch * bytes_per_param) / (intra_bw * 1e9 / 8)
                
                # DP cost: all-gather + reduce-scatter of params (overlapped with compute mostly)
                # But inter-node is slower
                dp_comm_cost = 0
                if dp > 1:
                    num_nodes_in_dp = max(1, dp * tp // gpus_per_node)
                    if num_nodes_in_dp > 1:
                        # Inter-node communication
                        dp_comm_cost = (model_memory_gb * 2) / (inter_bw / 8) * 0.3  # partially overlapped
                    else:
                        dp_comm_cost = (model_memory_gb * 2) / (intra_bw / 8) * 0.1
                
                # PP cost: pipeline bubble
                pp_bubble_cost = 0
                if pp > 1:
                    pp_bubble_cost = (pp - 1) * 0.5  # rough penalty per PP stage
                
                # Compute cost: inversely proportional to total_gpus
                compute_cost = (param_billions * batch_size * seq_len) / (total_gpus * 100)
                
                # Activation checkpointing recomputation penalty (~33% extra compute)
                if act_ckpt:
                    compute_cost *= 1.33
                
                # Inductor speedup (~10-15%)
                # Will be applied to all configs
                
                total_cost = compute_cost + tp_comm_cost + dp_comm_cost + pp_bubble_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_config = {
                        "tp": tp,
                        "dp": dp,
                        "pp": pp,
                        "activation_checkpointing": act_ckpt,
                    }
                
                break  # If no act_ckpt fits, use act_ckpt; don't need to try further
    
    if best_config is None:
        # Conservative fallback
        tp = pick_valid("tp", gpus_per_node)
        dp = pick_valid("dp", total_gpus // tp)
        pp = pick_valid("pp", 1)
        best_config = {
            "tp": tp, "dp": dp, "pp": pp,
            "activation_checkpointing": True,
        }
    
    tp = best_config["tp"]
    dp = best_config["dp"]
    pp = best_config["pp"]
    activation_checkpointing = best_config["activation_checkpointing"]
    
    # Micro batch: only matters for PP > 1
    if pp > 1:
        local_batch = batch_size // dp
        # More micro-batches = less pipeline bubble
        # Target smallest micro_batch that divides local_batch
        valid_mbs = sorted([m for m in mb_choices if local_batch % m == 0 and m <= local_batch])
        if valid_mbs:
            micro_batch = valid_mbs[0]  # smallest for most microbatches
        else:
            micro_batch = pick_valid("micro_batch", 1)
    else:
        micro_batch = pick_valid("micro_batch", 1)
    
    # Compile mode: inductor is generally faster for training
    compile_mode = "inductor" if "inductor" in valid.get("compile_mode", ["eager"]) else "eager"
    
    config = {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": compile_mode,
        "activation_checkpointing": activation_checkpointing,
    }
    
    return config
