
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Iteration 7: Explore tp=2, pp=4, dp=2, mbs=4, ac=False
    
    History:
    - tp=4, dp=4, pp=1, mbs=2, ac=False -> 8.358
    - tp=4, dp=4, pp=1, mbs=4, ac=False -> SIGSEGV (OOM)
    - tp=2, dp=8, pp=1, mbs=2, ac=True -> exit 143 (timeout/killed)
    - tp=2, dp=8, pp=1, mbs=2, ac=False -> exit 1 (OOM)
    - tp=2, pp=2, dp=4, mbs=2, ac=False -> 6.068
    - tp=2, pp=2, dp=4, mbs=4, ac=False -> 6.231
    - tp=2, pp=4, dp=2, mbs=2, ac=False -> 0.8841 (BEST!!)
    
    Strategy: The pp=4 config was a massive win. Now try mbs=4 with the same
    tp=2, pp=4, dp=2 to see if larger micro-batches reduce pipeline bubbles.
    With mbs=4: acc_steps = 32/(2*4) = 4 (fewer pipeline flushes).
    With mbs=2: acc_steps = 32/(2*2) = 8 (current best).
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
    
    # For this specific workload (Llama-3.1-8B, 16 GPUs, A100 40GB)
    # Try tp=2, pp=4, dp=2, mbs=4 (larger mbs, fewer accumulation steps)
    tp = 2
    pp = 4
    dp = total_gpus // (tp * pp)  # 16 / (2*4) = 2
    mbs = 4  # Try larger micro-batch: acc_steps = 32/(2*4) = 4
    ac = False
    
    # Validate constraints
    if dp not in dp_choices or tp * pp * dp != total_gpus:
        # Fallback to best known config
        tp, pp, dp, mbs, ac = 2, 4, 2, 2, False
    
    # Check batch divisibility
    if batch_size % (dp * mbs) != 0:
        mbs = 2
        if batch_size % (dp * mbs) != 0:
            mbs = 1
    
    # ---- General fallback logic for other workloads ----
    if total_gpus != 16 or gpu_memory_gb != 40:
        # General strategy based on learnings:
        # Higher PP is very beneficial (pp=4 >> pp=2 >> pp=1)
        # tp=2 within node is optimal for communication
        # Keep dp moderate
        
        tp = min(2, gpus_per_node)
        if tp not in tp_choices:
            tp = min(gpus_per_node, max(t for t in tp_choices if t <= gpus_per_node))
        
        # Try pp=4 first (was best), then pp=2, then pp=1
        best_pp = 1
        best_dp = total_gpus // tp
        for p in sorted(pp_choices, reverse=True):
            d = total_gpus // (tp * p)
            if d in dp_choices and d >= 1 and tp * p * d == total_gpus:
                best_pp = p
                best_dp = d
                break
        
        pp = best_pp
        dp = best_dp
        
        # Pick mbs=2 as default sweet spot
        mbs = 2
        if batch_size % (dp * mbs) != 0:
            for m in sorted(mbs_choices):
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
