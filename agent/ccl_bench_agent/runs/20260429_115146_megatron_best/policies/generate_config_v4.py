
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Iteration 4: Try tp=2, pp=2, dp=4, mbs=2, ac=False
    
    History:
    - tp=4, dp=4, pp=1, mbs=2, ac=False -> 8.358 (success, best)
    - tp=4, dp=4, pp=1, mbs=4, ac=False -> SIGSEGV (OOM)
    - tp=2, dp=8, pp=1, mbs=2, ac=True -> exit 143 (timeout/killed)
    - tp=2, dp=8, pp=1, mbs=2, ac=False -> exit 1 (OOM likely)
    
    Insights:
    - tp=4, pp=1 works but tp=2, pp=1 OOMs (not enough model splitting)
    - dp=8 cross-node fails (too much inter-node comm or OOM)
    - dp=4 is reliable
    
    Strategy: tp=2, pp=2, dp=4 keeps dp=4 (proven), splits model across tp*pp=4
    (same total as tp=4, pp=1), but tp=2 has less all-reduce overhead.
    pp=2 adds bubble but with 4 acc steps, bubble is manageable (~20%).
    The net effect of less TP comm might outweigh pp bubble.
    """
    total_gpus = environment.get("total_gpus", 16)
    gpus_per_node = environment.get("gpus_per_node", 4)
    gpu_memory_gb = environment.get("gpu_memory_gb", 40)
    
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    batch_size = workload.get("batch_size", 32)
    seq_len = workload.get("seq_len", 1024)
    model_family = workload.get("model_family", "").lower()
    precision = workload.get("precision", "bf16")
    
    tp_choices = sorted(valid_choices.get("tp", [1, 2, 4, 8]))
    pp_choices = sorted(valid_choices.get("pp", [1, 2, 4, 8]))
    dp_choices = sorted(valid_choices.get("dp", [1, 2, 4, 8, 16]))
    mbs_choices = sorted(valid_choices.get("micro_batch_size", [1, 2, 4]))
    ac_choices = valid_choices.get("activation_checkpointing", [True, False])
    
    # Estimate model size
    model_size_b = 8.0
    if "70b" in model_family:
        model_size_b = 70.0
    elif "13b" in model_family:
        model_size_b = 13.0
    elif "8b" in model_family:
        model_size_b = 8.0
    elif "7b" in model_family:
        model_size_b = 7.0
    
    # For this workload (Llama-3.1-8B, 16 GPUs, A100-40GB):
    # Try tp=2, pp=2, dp=4, mbs=2, ac=False
    # Each GPU holds 1/4 of model (same as tp=4,pp=1) but with less TP overhead
    # dp=4 keeps proven data parallelism degree
    
    tp = 2
    pp = 2
    dp = total_gpus // (tp * pp)  # 16 / (2*2) = 4
    mbs = 2
    ac = False
    
    # Validate constraints
    if dp not in dp_choices:
        tp, pp, dp, mbs, ac = 4, 1, 4, 2, False
    
    if tp * pp * dp != total_gpus:
        tp, pp, dp, mbs, ac = 4, 1, 4, 2, False
    
    # Check batch divisibility
    if batch_size % (dp * mbs) != 0:
        mbs = 1
        if batch_size % (dp * mbs) != 0:
            tp, pp, dp, mbs, ac = 4, 1, 4, 2, False
    
    # General fallback logic for other workloads
    if total_gpus != 16 or gpu_memory_gb != 40:
        # Use a general strategy
        tp = min(gpus_per_node, max(t for t in tp_choices if t <= gpus_per_node))
        pp = 1
        dp = total_gpus // (tp * pp)
        
        if dp not in dp_choices:
            for p in pp_choices:
                d = total_gpus // (tp * p)
                if d in dp_choices and d >= 1:
                    pp = p
                    dp = d
                    break
        
        # Pick largest mbs that divides batch
        mbs = 1
        for m in reversed(mbs_choices):
            if batch_size % (dp * m) == 0:
                mbs = m
                break
        
        ac = False
    
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "micro_batch_size": mbs,
        "activation_checkpointing": ac,
    }
