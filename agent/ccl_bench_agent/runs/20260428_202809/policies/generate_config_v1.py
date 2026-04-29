
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    gpu_memory_gb = environment.get("gpu_memory_gb", 80)
    batch_size = workload.get("batch_size", 1)
    seq_len = workload.get("seq_len", 2048)
    precision = workload.get("precision", "bf16")
    
    # Build lookup of valid choices per key
    config_space = workload.get("config_space", [])
    valid = {}
    for dim in config_space:
        valid[dim["key"]] = dim["choices"]
    
    # Model size heuristic based on model_family
    model_family = workload.get("model_family", "")
    
    # Estimate model size in GB (rough: params * bytes_per_param)
    # Llama-8B ~ 16GB in fp32, ~8GB in bf16
    bytes_per_param = 2 if precision in ("bf16", "fp16") else 4
    
    # For 8B-class models on 40GB A100, tp=2 should suffice
    # For larger models, we need more tp
    if gpu_memory_gb >= 80:
        # A100 80GB or H100 - can fit more per GPU
        if "70b" in model_family.lower() or "65b" in model_family.lower():
            tp = 8
        elif "13b" in model_family.lower():
            tp = 2
        else:
            tp = 1  # 8B fits on 80GB
    elif gpu_memory_gb >= 40:
        # A100 40GB
        if "70b" in model_family.lower() or "65b" in model_family.lower():
            tp = 8
        elif "13b" in model_family.lower():
            tp = 4
        else:
            tp = 2  # 8B needs tp=2 on 40GB for training (model + optimizer + activations)
    else:
        tp = min(gpus_per_node, 8)
    
    # Clamp tp to valid choices and available GPUs
    if "tp" in valid:
        tp_choices = sorted(valid["tp"])
        # Pick the closest valid choice >= our target
        tp = min([c for c in tp_choices if c >= tp], default=tp_choices[-1])
        tp = min(tp, total_gpus)
        if tp not in tp_choices:
            tp = max([c for c in tp_choices if c <= tp], default=tp_choices[0])
    
    # Keep tp within a single node for NVLink benefit
    tp = min(tp, gpus_per_node)
    
    # Remaining GPUs for dp and pp
    remaining = total_gpus // tp
    
    # For most training workloads, pp=1 is best unless model is very large
    pp = 1
    dp = remaining // pp
    
    # Validate dp
    if "dp" in valid:
        dp_choices = sorted(valid["dp"])
        if dp not in dp_choices:
            dp = max([c for c in dp_choices if c <= dp], default=dp_choices[0])
    
    # Ensure tp * dp * pp = total_gpus (or as close as possible)
    # If not, try adjusting
    if tp * dp * pp != total_gpus:
        # Try pp > 1
        for pp_candidate in sorted(valid.get("pp", [1])):
            dp_candidate = total_gpus // (tp * pp_candidate)
            if dp_candidate >= 1 and tp * dp_candidate * pp_candidate == total_gpus:
                if "dp" not in valid or dp_candidate in valid["dp"]:
                    dp = dp_candidate
                    pp = pp_candidate
                    break
    
    # micro_batch: must divide local_batch_size = batch_size / dp
    local_batch = batch_size // dp if dp > 0 else batch_size
    
    if "micro_batch" in valid:
        mb_choices = sorted(valid["micro_batch"], reverse=True)
        # Pick largest micro_batch that divides local_batch
        micro_batch = 1
        for mb in mb_choices:
            if mb <= local_batch and local_batch % mb == 0:
                micro_batch = mb
                break
        # If none divides, pick the largest that's <= local_batch
        if micro_batch == 1 and local_batch > 1:
            for mb in mb_choices:
                if mb <= local_batch:
                    micro_batch = mb
                    break
    else:
        micro_batch = max(1, local_batch)
    
    # compile_mode: inductor is generally faster for training
    compile_mode = "inductor" if "compile_mode" in valid and "inductor" in valid["compile_mode"] else "eager"
    
    # Activation checkpointing: enable if memory is tight
    # For 8B on 40GB with tp=2, it might be needed but adds overhead
    # Let's try without first for speed
    activation_checkpointing = False
    
    config = {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": compile_mode,
        "activation_checkpointing": activation_checkpointing,
    }
    
    return config
