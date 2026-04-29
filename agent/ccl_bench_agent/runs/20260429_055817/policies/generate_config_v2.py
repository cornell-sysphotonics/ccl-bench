
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

    def pick_closest(choices, target):
        """Pick the closest valid choice to target."""
        return min(choices, key=lambda x: abs(x - target))

    def pick_valid(key, target):
        """Pick closest valid choice for a key."""
        choices = valid.get(key, [target])
        return min(choices, key=lambda x: abs(x - target))

    # Estimate model size roughly
    # Common model sizes: 8B ~ 16GB bf16, 70B ~ 140GB bf16
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
        param_billions = 8  # default guess

    bytes_per_param = 2 if precision in ["bf16", "fp16"] else 4
    model_memory_gb = param_billions * bytes_per_param  # just params
    # Training memory per GPU with FSDP: params/tp + optimizer_states/(tp*dp) + activations
    # optimizer states ~= 12 bytes/param for Adam (param + grad + 2 states in fp32)
    # With activation checkpointing, activation memory is much reduced

    # Strategy: minimize TP (communication overhead) while fitting in memory
    # FSDP shards optimizer states across dp ranks
    # TP shards model params and reduces activation memory per GPU
    
    # For each valid TP, compute if memory fits, then maximize DP
    # Lower TP = less communication overhead = faster step time
    # But need enough TP to fit in GPU memory
    
    tp_choices = sorted(valid.get("tp", [1]))
    dp_choices = sorted(valid.get("dp", [1]), reverse=True)
    pp_choices = sorted(valid.get("pp", [1]))
    
    best_config = None
    
    for tp in tp_choices:
        # TP must not exceed gpus_per_node for NVLink efficiency
        if tp > gpus_per_node:
            continue
            
        remaining_gpus = total_gpus // tp
        
        for dp in dp_choices:
            if dp > remaining_gpus:
                continue
            
            pp_remaining = remaining_gpus // dp
            # Prefer pp=1
            if pp_remaining >= 1:
                pp = pick_valid("pp", 1) if pp_remaining >= 1 else pick_valid("pp", pp_remaining)
                
                if tp * dp * pp > total_gpus:
                    continue
                if pp not in valid.get("pp", [1]):
                    continue
                    
                # Check memory feasibility (rough estimate)
                # Per-GPU param memory with TP
                param_mem_per_gpu = model_memory_gb / tp  # params sharded by TP
                
                # Optimizer states: 12 bytes/param total, sharded by dp*tp
                opt_mem_per_gpu = (param_billions * 12) / (tp * dp)
                
                # Activation memory (rough): batch_per_gpu * seq_len * hidden_dim * num_layers * 2bytes
                # With activation checkpointing, this is greatly reduced (sqrt layers)
                local_batch = batch_size // dp
                if local_batch < 1:
                    continue
                    
                # Very rough activation estimate
                # hidden_dim ~ param_billions * 1024 / num_layers (very rough)
                hidden_dim = int((param_billions * 1e9 / (num_layers * 12)) ** 0.5) if num_layers > 0 else 4096
                hidden_dim = min(hidden_dim, 8192)
                
                act_mem_per_gpu = (local_batch * seq_len * hidden_dim * num_layers * 2) / (tp * 1e9)
                # With activation checkpointing, reduce by ~5x
                act_mem_checkpointed = act_mem_per_gpu / 5
                
                total_mem = param_mem_per_gpu + opt_mem_per_gpu + act_mem_checkpointed
                
                # Add some buffer (fragmentation, etc.)
                mem_limit = gpu_memory_gb * 0.85
                
                needs_act_ckpt = (param_mem_per_gpu + opt_mem_per_gpu + act_mem_per_gpu) > mem_limit
                
                if needs_act_ckpt:
                    total_mem_final = param_mem_per_gpu + opt_mem_per_gpu + act_mem_checkpointed
                    act_ckpt = True
                else:
                    total_mem_final = param_mem_per_gpu + opt_mem_per_gpu + act_mem_per_gpu
                    act_ckpt = False
                
                if total_mem_final > mem_limit:
                    # Still doesn't fit, need more TP or activation checkpointing
                    # Try with activation checkpointing
                    total_mem_final = param_mem_per_gpu + opt_mem_per_gpu + act_mem_checkpointed
                    act_ckpt = True
                    if total_mem_final > mem_limit:
                        continue
                
                # This config fits! 
                # Prefer: lower TP (less comm), higher DP, pp=1
                # Score heuristic: lower TP is better, pp=1 is best
                if best_config is None:
                    best_config = {
                        "tp": tp, "dp": dp, "pp": pp, 
                        "activation_checkpointing": act_ckpt,
                        "local_batch": local_batch
                    }
                    break  # We found the best dp for this tp
        
        if best_config is not None:
            break  # Found a config with minimum TP
    
    # Fallback if no config found
    if best_config is None:
        # Conservative fallback: use all GPUs with TP=gpus_per_node
        tp = pick_valid("tp", gpus_per_node)
        dp = pick_valid("dp", total_gpus // tp)
        pp = pick_valid("pp", 1)
        best_config = {
            "tp": tp, "dp": dp, "pp": pp,
            "activation_checkpointing": True,
            "local_batch": max(1, batch_size // dp)
        }
    
    tp = best_config["tp"]
    dp = best_config["dp"]
    pp = best_config["pp"]
    activation_checkpointing = best_config["activation_checkpointing"]
    
    # Micro batch: only matters for PP > 1
    if pp > 1:
        local_batch = batch_size // dp
        # More micro-batches reduces pipeline bubble
        # Target: num_microbatches >= 2*pp
        mb_target = max(1, local_batch // (2 * pp))
        micro_batch = pick_valid("micro_batch", mb_target)
        # Ensure micro_batch divides local_batch if possible
        mb_choices = valid.get("micro_batch", [1])
        valid_mbs = [m for m in mb_choices if local_batch % m == 0 and m <= local_batch]
        if valid_mbs:
            # Pick smallest valid micro_batch for most pipeline stages
            micro_batch = min(valid_mbs)
    else:
        micro_batch = pick_valid("micro_batch", 1)
    
    # Compile mode: inductor is generally faster
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
