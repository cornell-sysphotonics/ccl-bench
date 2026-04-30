
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Iteration 5: Try tp=2, pp=2, dp=4, mbs=4, ac=False
    
    History:
    - tp=4, dp=4, pp=1, mbs=2, ac=False -> 8.358 (success)
    - tp=4, dp=4, pp=1, mbs=4, ac=False -> SIGSEGV (OOM)
    - tp=2, dp=8, pp=1, mbs=2, ac=True -> exit 143 (timeout/killed)
    - tp=2, dp=8, pp=1, mbs=2, ac=False -> exit 1 (OOM)
    - tp=2, pp=2, dp=4, mbs=2, ac=False -> 6.068 (BEST)
    
    Strategy: Keep the best parallelism config (tp=2, pp=2, dp=4) but try mbs=4.
    With pp=2, each pipeline stage holds half the layers, so less activation memory.
    mbs=4 would give acc_steps=32/(4*4)=2, reducing gradient accumulation overhead.
    If OOM, we'll know mbs=4 doesn't work even with pp=2.
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
    
    # For this specific workload, try mbs=4 with the best parallelism config
    tp = 2
    pp = 2
    dp = total_gpus // (tp * pp)  # 16 / (2*2) = 4
    mbs = 4
    ac = False
    
    # Validate constraints
    if dp not in dp_choices:
        tp, pp, dp, mbs, ac = 2, 2, 4, 2, False
    
    if tp * pp * dp != total_gpus:
        tp, pp, dp, mbs, ac = 2, 2, 4, 2, False
    
    # Check batch divisibility
    if batch_size % (dp * mbs) != 0:
        mbs = 2
        if batch_size % (dp * mbs) != 0:
            mbs = 1
    
    # General fallback logic for other workloads
    if total_gpus != 16 or gpu_memory_gb != 40:
        # General strategy: TP within node, add PP if needed, rest is DP
        # Start with tp=2 which has good balance of memory reduction vs comm
        tp = min(2, gpus_per_node)
        if tp not in tp_choices:
            tp = min(gpus_per_node, max(t for t in tp_choices if t <= gpus_per_node))
        
        # Try pp=2 first since it helped in our benchmarks
        pp = 2 if 2 in pp_choices else 1
        dp = total_gpus // (tp * pp)
        
        if dp not in dp_choices or dp < 1:
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
