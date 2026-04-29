
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Optimized config for LLM training on multi-node GPU clusters.
    
    Key insights from history:
    - tp=4, dp=4, pp=1, mbs=2, no checkpoint: score 8.697 (success)
    - tp=2, dp=8, pp=1, mbs=2, checkpoint: FAILED (OOM or constraint violation)
    - Llama-8B on A100-40GB needs tp>=4
    
    Strategy: Keep tp=4 (within node for fast communication), maximize micro_batch
    to reduce gradient accumulation steps and overhead.
    """
    total_gpus = environment.get("total_gpus", 16)
    gpus_per_node = environment.get("gpus_per_node", 4)
    gpu_memory_gb = environment.get("gpu_memory_gb", 40)
    inter_bw = environment.get("inter_node_bandwidth_gbps", 25)
    intra_bw = environment.get("intra_node_bandwidth_gbps", 600)
    
    # Extract config space to know valid choices
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    batch_size = workload.get("batch_size", 16)
    seq_len = workload.get("seq_len", 1024)
    precision = workload.get("precision", "bf16")
    model_family = workload.get("model_family", "").lower()
    
    # Determine model size category
    is_large_model = any(x in model_family for x in ["70b", "65b", "40b", "34b"])
    is_medium_model = any(x in model_family for x in ["8b", "7b", "13b"])
    
    # For medium models (7B-13B) on A100-40GB: tp=4 works, tp=2 may OOM
    # For large models: need more TP or PP
    
    if is_large_model:
        tp = min(8, gpus_per_node)
        pp = 2
    else:
        # Medium or small models: tp within a node
        tp = min(4, gpus_per_node)
        pp = 1
    
    # Ensure tp is valid
    tp_choices = valid_choices.get("tp", [1, 2, 4, 8])
    if tp not in tp_choices:
        tp = max(c for c in tp_choices if c <= tp) if any(c <= tp for c in tp_choices) else min(tp_choices)
    
    pp_choices = valid_choices.get("pp", [1, 2, 4, 8])
    if pp not in pp_choices:
        pp = max(c for c in pp_choices if c <= pp) if any(c <= pp for c in pp_choices) else min(pp_choices)
    
    # Compute dp
    dp = total_gpus // (tp * pp)
    dp_choices = valid_choices.get("dp", [1, 2, 4, 8, 16])
    
    # If dp isn't valid, adjust pp
    if dp not in dp_choices:
        # Try different pp values
        for pp_try in sorted(pp_choices):
            dp_try = total_gpus // (tp * pp_try)
            if dp_try in dp_choices and dp_try >= 1:
                pp = pp_try
                dp = dp_try
                break
    
    # If still invalid, try different tp values
    if dp not in dp_choices:
        for tp_try in sorted(tp_choices, reverse=True):
            for pp_try in sorted(pp_choices):
                dp_try = total_gpus // (tp_try * pp_try)
                if dp_try in dp_choices and dp_try >= 1:
                    tp = tp_try
                    pp = pp_try
                    dp = dp_try
                    break
            if dp in dp_choices:
                break
    
    # Determine micro_batch_size
    # Want largest micro_batch that divides samples_per_rank = batch_size / dp
    mbs_choices = sorted(valid_choices.get("micro_batch_size", [1, 2, 4]), reverse=True)
    
    if dp > 0:
        samples_per_rank = batch_size // dp
    else:
        samples_per_rank = batch_size
    
    micro_batch = 1
    for mb in mbs_choices:
        if samples_per_rank >= mb and samples_per_rank % mb == 0:
            micro_batch = mb
            break
    
    # Activation checkpointing: disable for speed unless memory is very tight
    # On A100-40GB with tp=4, Llama-8B fits without checkpointing (confirmed by run 1)
    # With larger micro_batch, memory usage increases, so be cautious
    # For micro_batch=4, seq=1024, activations may be larger - let's try without first
    activation_checkpointing = False
    
    # If we're pushing micro_batch high on limited memory, enable checkpointing
    if gpu_memory_gb <= 40 and micro_batch >= 4 and seq_len >= 2048:
        activation_checkpointing = True
    
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": activation_checkpointing,
    }
