
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Iteration 6: Explore tp=2, pp=4, dp=2, mbs=2, ac=False
    
    History:
    - tp=4, dp=4, pp=1, mbs=2, ac=False -> 8.358
    - tp=4, dp=4, pp=1, mbs=4, ac=False -> SIGSEGV (OOM)
    - tp=2, dp=8, pp=1, mbs=2, ac=True -> exit 143 (timeout/killed)
    - tp=2, dp=8, pp=1, mbs=2, ac=False -> exit 1 (OOM)
    - tp=2, pp=2, dp=4, mbs=2, ac=False -> 6.068 (BEST)
    - tp=2, pp=2, dp=4, mbs=4, ac=False -> 6.231
    
    Strategy: Try more pipeline parallelism (pp=4) with same tp=2.
    pp=4 means each stage has only 8 layers (32/4), significantly less memory.
    dp=2 means acc_steps = 32/(2*2) = 8.
    More PP stages = more pipeline bubble, but each stage is lighter.
    This tests whether the reduced memory/computation per stage can offset
    the pipeline bubble overhead.
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
    
    # For this specific workload (Llama-3.1-8B, 16 GPUs, A100 40GB), explore pp=4
    tp = 2
    pp = 4
    dp = total_gpus // (tp * pp)  # 16 / (2*4) = 2
    mbs = 2
    ac = False
    
    # Validate constraints
    if dp not in dp_choices or tp * pp * dp != total_gpus:
        # Fallback to best known config
        tp, pp, dp, mbs, ac = 2, 2, 4, 2, False
    
    # Check batch divisibility
    if batch_size % (dp * mbs) != 0:
        mbs = 2
        if batch_size % (dp * mbs) != 0:
            mbs = 1
    
    # ---- General fallback logic for other workloads ----
    if total_gpus != 16 or gpu_memory_gb != 40:
        # General strategy based on learnings:
        # tp=2 within node is optimal, pp=2 helps with memory, maximize dp
        tp = min(2, gpus_per_node)
        if tp not in tp_choices:
            tp = min(gpus_per_node, max(t for t in tp_choices if t <= gpus_per_node))
        
        # pp=2 was beneficial - try it first
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
        
        # Pick mbs=2 as sweet spot (mbs=4 was slightly worse, mbs=1 more accumulation)
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
