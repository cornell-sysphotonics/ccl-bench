
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    gpu_memory_gb = environment.get("gpu_memory_gb", 80)
    batch_size = workload.get("batch_size", 1)
    seq_len = workload.get("seq_len", 1024)
    model_family = workload.get("model_family", "")
    precision = workload.get("precision", "fp32")

    # Build lookup of valid choices per key
    config_space = workload.get("config_space", [])
    valid = {}
    for dim in config_space:
        valid[dim["key"]] = dim["choices"]

    # Estimate model size in GB (rough)
    # Llama-3.1-8B ~ 8B params, bf16 = 2 bytes/param = ~16GB just for params
    # With optimizer states (Adam: 2x fp32 copies) ~ 48GB for 8B model
    # We need to shard across GPUs

    # Strategy: 
    # 1. Use TP within node for fast comm (NVLink)
    # 2. Use DP (FSDP) across nodes to shard optimizer states
    # 3. Avoid PP if possible (pipeline bubbles)

    def pick_closest(choices, target):
        """Pick the closest valid choice to target."""
        return min(choices, key=lambda x: abs(x - target))

    # Determine TP: use full node width for large models, less for smaller ones
    # For 8B model on 40GB GPUs, TP=2 might suffice per node, but TP=4 is safer
    if "8B" in str(model_family) or "8b" in str(model_family):
        tp_target = min(gpus_per_node, 4)  # TP=4 within node
    elif "70B" in str(model_family) or "70b" in str(model_family):
        tp_target = gpus_per_node  # Full node for large models
    else:
        # Default: estimate based on memory
        bytes_per_param = 2 if precision in ["bf16", "fp16"] else 4
        # Rough param count heuristic - not critical, we'll be conservative
        tp_target = min(gpus_per_node, 4)

    tp = pick_closest(valid.get("tp", [1]), tp_target)

    # Remaining GPUs for DP * PP
    remaining = total_gpus // tp

    # Prefer PP=1, use all remaining for DP (FSDP shards optimizer states)
    pp = 1
    dp = remaining

    # Validate against choices
    if dp not in valid.get("dp", [1]):
        # Try PP > 1 to find valid combination
        best_dp, best_pp = 1, 1
        best_total = 0
        for pp_c in valid.get("pp", [1]):
            for dp_c in valid.get("dp", [1]):
                if dp_c * pp_c * tp <= total_gpus and dp_c * pp_c * tp > best_total:
                    best_dp, best_pp = dp_c, pp_c
                    best_total = dp_c * pp_c * tp
        dp, pp = best_dp, best_pp
    
    pp = pick_closest(valid.get("pp", [1]), pp)
    dp = pick_closest(valid.get("dp", [1]), dp)

    # Ensure tp * dp * pp <= total_gpus
    while tp * dp * pp > total_gpus:
        if dp > 1:
            dp_choices = [c for c in valid.get("dp", [1]) if c < dp]
            if dp_choices:
                dp = max(dp_choices)
            else:
                break
        elif pp > 1:
            pp_choices = [c for c in valid.get("pp", [1]) if c < pp]
            if pp_choices:
                pp = max(pp_choices)
            else:
                break
        else:
            break

    # Micro batch: for PP > 1, more micro-batches reduce bubble ratio
    # For PP=1, it doesn't matter much
    if pp > 1:
        # More micro-batches = less bubble. Target: at least 2*pp micro-batches
        local_batch = batch_size // dp
        # micro_batch * num_microbatches = local_batch
        # We want num_microbatches to be large, so micro_batch should be small but not too small
        mb_target = max(1, local_batch // (2 * pp))
        micro_batch = pick_closest(valid.get("micro_batch", [1]), mb_target)
    else:
        micro_batch = pick_closest(valid.get("micro_batch", [1]), 1)

    # Activation checkpointing: enable for memory-constrained setups
    # With 40GB A100 and 8B model, we likely need it
    activation_checkpointing = True
    if "activation_checkpointing" in valid:
        activation_checkpointing = True if True in valid["activation_checkpointing"] else False

    # Compile mode: inductor generally faster for training
    compile_mode = "inductor" if "inductor" in valid.get("compile_mode", ["eager"]) else "eager"

    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": compile_mode,
        "activation_checkpointing": activation_checkpointing,
    }
