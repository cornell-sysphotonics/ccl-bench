
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
        if isinstance(target, bool):
            return target if target in choices else choices[0]
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

    intra_bw = environment.get("intra_node_bandwidth_gbps", 600)
    inter_bw = environment.get("inter_node_bandwidth_gbps", 100)

    # For this specific scenario (Llama-3.1-8B, 16 A100-40GB, 4/node):
    # Best so far: tp=4, dp=4, pp=1, inductor, act_ckpt=True → 8.817
    # tp=2, dp=8 and tp=1, dp=16 both gave 10.94 (without act_ckpt)
    #
    # Strategy: tp=4, dp=4 is clearly best parallelism config.
    # Try WITHOUT activation checkpointing to save recomputation overhead.
    # Memory estimate with tp=4, dp=4:
    #   Params: 16GB / 4(tp) = 4GB per GPU
    #   Optimizer (Adam fp32): 8B * 12 bytes / (4*4) = 6GB per GPU  
    #   Gradients: 4GB per GPU
    #   Activations (batch=8, seq=1024, hidden=4096, 32 layers, /tp=4):
    #     ~8 * 1024 * 4096 * 32 * 2 * 2 / 4 / 1e9 ≈ ~17GB without checkpointing
    #   Total: ~31GB - might be tight on 40GB but worth trying
    
    # General algorithm for choosing parallelism
    num_nodes = total_gpus // gpus_per_node
    
    # Determine hidden_dim from model
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

    # Try configurations in order of expected performance
    # Prefer: tp within node, dp across nodes, pp=1
    candidates = []
    
    tp_choices = sorted(valid.get("tp", [1]))
    pp_choices = sorted(valid.get("pp", [1]))
    
    for tp in tp_choices:
        if tp > gpus_per_node:
            continue
        for pp in pp_choices:
            dp_needed = total_gpus // (tp * pp)
            if dp_needed < 1 or dp_needed not in valid.get("dp", [1]):
                continue
            if tp * dp_needed * pp != total_gpus:
                continue
            dp = dp_needed
            
            local_batch = batch_size // dp
            if local_batch < 1:
                continue
            
            # Memory estimation per GPU
            param_mem = model_memory_gb / tp
            # Optimizer states: Adam has 2 fp32 copies + gradients
            # Total ~12 bytes per param for bf16 training with Adam
            opt_mem = (param_billions * 12) / (tp * dp)  # FSDP shards across dp
            grad_mem = model_memory_gb / (tp * dp)  # FSDP shards gradients too
            
            # Activation memory (very rough)
            act_per_layer = (local_batch * seq_len * hidden_dim * bytes_per_param * 4) / (tp * 1e9)
            act_mem_full = act_per_layer * num_layers
            act_mem_ckpt = act_per_layer * 2  # Only keep ~2 layers with checkpointing
            
            for act_ckpt in [False, True]:
                act_mem = act_mem_ckpt if act_ckpt else act_mem_full
                total_mem = param_mem + opt_mem + grad_mem + act_mem
                
                if total_mem > gpu_memory_gb * 0.9:
                    continue
                
                # Cost model
                # Compute time (base)
                compute = (param_billions * local_batch * seq_len * 6) / (tp * 1e12)
                if act_ckpt:
                    compute *= 1.33  # recomputation overhead
                
                # TP communication cost (all-reduce, within node via NVLink)
                tp_comm = 0
                if tp > 1:
                    msg_size = 2 * local_batch * seq_len * hidden_dim * bytes_per_param / 1e9
                    # Ring all-reduce: 2*(tp-1)/tp * msg_size
                    tp_comm = 2 * (tp - 1) / tp * msg_size * num_layers * 2 / (intra_bw / 8)
                
                # DP communication cost (FSDP all-gather + reduce-scatter)
                dp_comm = 0
                if dp > 1:
                    dp_msg = model_memory_gb / tp  # params to gather/scatter
                    nodes_in_dp = max(1, dp // (gpus_per_node // tp))
                    if nodes_in_dp > 1:
                        effective_bw = inter_bw / 8  # GB/s
                        dp_comm = dp_msg * 2 * 2 * (dp - 1) / dp / effective_bw
                        dp_comm *= 0.3  # overlap with compute
                    else:
                        dp_comm = dp_msg * 2 * 2 * (dp - 1) / dp / (intra_bw / 8)
                        dp_comm *= 0.1  # mostly overlapped within node
                
                # PP bubble cost
                pp_cost = 0
                if pp > 1:
                    pp_cost = compute * (pp - 1) / pp  # bubble fraction
                
                total_cost = compute + tp_comm + dp_comm + pp_cost
                
                candidates.append({
                    "tp": tp, "dp": dp, "pp": pp,
                    "activation_checkpointing": act_ckpt,
                    "cost": total_cost,
                    "mem": total_mem,
                })
    
    # Sort by cost
    candidates.sort(key=lambda x: x["cost"])
    
    if candidates:
        best = candidates[0]
    else:
        # Fallback: safe config
        best = {
            "tp": pick_valid("tp", gpus_per_node),
            "dp": pick_valid("dp", total_gpus // gpus_per_node),
            "pp": pick_valid("pp", 1),
            "activation_checkpointing": True,
        }
    
    tp = best["tp"]
    dp = best["dp"]
    pp = best["pp"]
    activation_checkpointing = best["activation_checkpointing"]
    
    # Micro batch: only matters for PP > 1
    if pp > 1:
        local_batch = batch_size // dp
        mb_choices = valid.get("micro_batch", [1])
        valid_mbs = sorted([m for m in mb_choices if local_batch % m == 0 and m <= local_batch])
        if valid_mbs:
            micro_batch = valid_mbs[0]
        else:
            micro_batch = pick_valid("micro_batch", 1)
    else:
        micro_batch = pick_valid("micro_batch", 1)
    
    # Compile mode: inductor generally faster
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
