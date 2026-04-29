
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
    num_nodes = total_gpus // gpus_per_node

    # Hidden dim estimation
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

    # Strategy based on execution history:
    # Best: tp=4, dp=4, pp=1, inductor, act_ckpt=True → 8.817
    # tp=4, dp=4, pp=1, inductor, act_ckpt=False → 10.94 (worse! maybe OOM-adjacent or other issue)
    # 
    # Now try: tp=2, dp=8, pp=1, inductor, act_ckpt=True
    # Less TP overhead (2 vs 4), more DP but FSDP overlaps well
    # local_batch = 32/8 = 4 (smaller, less activation memory)
    # DP goes inter-node but bandwidth is 200 Gbps which is decent
    
    # Ensure tp * dp * pp = total_gpus and all values are valid
    tp_choices = sorted(valid.get("tp", [1]))
    dp_choices = sorted(valid.get("dp", [1]))
    pp_choices = sorted(valid.get("pp", [1]))
    mb_choices = sorted(valid.get("micro_batch", [1]))
    
    # Generate candidate configs and pick best based on cost model
    candidates = []
    
    for tp in tp_choices:
        if tp > gpus_per_node:
            continue  # TP should stay within node for NVLink
        # num_heads must be divisible by tp
        if num_heads % tp != 0:
            continue
        for pp in pp_choices:
            dp_needed = total_gpus // (tp * pp)
            if dp_needed < 1 or dp_needed not in dp_choices:
                continue
            if tp * dp_needed * pp != total_gpus:
                continue
            dp = dp_needed

            local_batch = batch_size // dp
            if local_batch < 1:
                continue

            # Memory estimation per GPU
            # FSDP shards parameters and optimizer states across dp dimension
            # During forward: all-gather brings full params (sharded by tp)
            # Peak param memory = model_memory_gb / tp (full params after all-gather)
            param_mem = model_memory_gb / tp
            
            # Optimizer states: sharded across dp
            # Adam: 12 bytes/param (master fp32 weights + momentum + variance)
            opt_mem = (param_billions * 12) / (tp * dp)
            # Gradients: 2 bytes/param, sharded by dp after reduce-scatter
            grad_mem = (param_billions * bytes_per_param) / (tp * dp)
            
            # Activation memory
            # Per layer: roughly batch * seq * hidden * factor / tp
            act_per_layer = (10 * local_batch * seq_len * hidden_dim * bytes_per_param) / (tp * 1e9)
            eff_layers = num_layers if pp == 1 else num_layers // pp

            for act_ckpt in [False, True]:
                if act_ckpt:
                    # With checkpointing: only ~2 layers of activations at a time
                    act_mem = act_per_layer * 2
                else:
                    act_mem = act_per_layer * eff_layers
                
                total_mem = param_mem + opt_mem + grad_mem + act_mem + 3.0  # 3GB overhead
                
                if total_mem > gpu_memory_gb * 0.90:
                    continue

                # Cost model
                # Compute cost (proportional to FLOPs per GPU)
                compute_base = (param_billions * local_batch * seq_len * 6) / (tp * 1e12)
                if pp > 1:
                    compute_base /= pp
                recomp_factor = 1.33 if act_ckpt else 1.0
                compute = compute_base * recomp_factor

                # TP communication cost
                tp_comm = 0
                if tp > 1:
                    # 2 all-reduces per layer (forward + backward), each with msg_size
                    msg_size = 2 * local_batch * seq_len * hidden_dim * bytes_per_param / 1e9
                    tp_comm = 4 * (tp - 1) / tp * msg_size * eff_layers / (intra_bw / 8)

                # DP (FSDP) communication cost
                # All-gather in forward + reduce-scatter in backward
                dp_comm = 0
                if dp > 1:
                    dp_msg = model_memory_gb / tp  # total param data to communicate
                    # Determine effective bandwidth
                    gpus_in_dp_same_node = gpus_per_node // tp
                    if dp > gpus_in_dp_same_node:
                        # Inter-node communication dominates
                        effective_bw = inter_bw / 8  # GB/s
                        dp_comm = dp_msg * 2 * (dp - 1) / dp / effective_bw
                        # FSDP can overlap with compute
                        dp_comm *= 0.4
                    else:
                        effective_bw = intra_bw / 8
                        dp_comm = dp_msg * 2 * (dp - 1) / dp / effective_bw
                        dp_comm *= 0.15

                # PP bubble cost
                pp_cost = 0
                if pp > 1:
                    valid_mbs = sorted([m for m in mb_choices if local_batch % m == 0 and m <= local_batch])
                    if valid_mbs:
                        best_mb = valid_mbs[0]
                    else:
                        best_mb = 1
                    num_microbatches = local_batch // best_mb
                    bubble_frac = (pp - 1) / (num_microbatches + pp - 1)
                    pp_cost = compute * bubble_frac * pp
                    pp_p2p = 2 * local_batch * seq_len * hidden_dim * bytes_per_param / (tp * 1e9)
                    if num_nodes > 1:
                        pp_cost += pp_p2p / (inter_bw / 8)
                    else:
                        pp_cost += pp_p2p / (intra_bw / 8)

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
        # Fallback
        best = {
            "tp": pick_valid("tp", min(gpus_per_node, 4)),
            "dp": pick_valid("dp", max(1, total_gpus // min(gpus_per_node, 4))),
            "pp": pick_valid("pp", 1),
            "activation_checkpointing": True,
        }

    tp = best["tp"]
    dp = best["dp"]
    pp = best["pp"]
    activation_checkpointing = best["activation_checkpointing"]

    # Micro batch
    if pp > 1:
        local_batch = batch_size // dp
        valid_mbs = sorted([m for m in mb_choices if local_batch % m == 0 and m <= local_batch])
        if valid_mbs:
            micro_batch = valid_mbs[0]
        else:
            micro_batch = pick_valid("micro_batch", 1)
    else:
        micro_batch = pick_valid("micro_batch", 1)

    # Compile mode
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
