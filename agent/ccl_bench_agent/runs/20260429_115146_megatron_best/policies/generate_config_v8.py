
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Iteration 8: Explore tp=1, pp=4, dp=4, mbs=2, ac=False
    
    History:
    - tp=4, dp=4, pp=1, mbs=2, ac=False -> 8.358
    - tp=4, dp=4, pp=1, mbs=4, ac=False -> SIGSEGV (OOM)
    - tp=2, dp=8, pp=1, mbs=2, ac=True -> exit 143 (timeout/killed)
    - tp=2, dp=8, pp=1, mbs=2, ac=False -> exit 1 (OOM)
    - tp=2, pp=2, dp=4, mbs=2, ac=False -> 6.068
    - tp=2, pp=2, dp=4, mbs=4, ac=False -> 6.231
    - tp=2, pp=4, dp=2, mbs=2, ac=False -> 0.8841 (BEST!!)
    - tp=2, pp=4, dp=2, mbs=4, ac=False -> 1.027
    
    Key learnings:
    - pp=4 is hugely beneficial (0.88 vs 6.07 vs 8.36)
    - mbs=2 beats mbs=4 at dp=2,pp=4
    - tp=2 works well, but tp=1 with pp=4 might be even better (no TP comm)
    - With pp=4, model split into 4 stages, ~2B params per stage = ~4GB bf16
      Plus optimizer ~12GB + activations should fit 40GB even with tp=1
    
    Try: tp=1, pp=4, dp=4, mbs=2 -> acc_steps = 32/(4*2) = 4
    This halves accumulation steps vs best config (was 8), and eliminates TP overhead.
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
    
    # Primary exploration: tp=1, pp=4, dp=4, mbs=2
    tp = 1
    pp = 4
    dp = total_gpus // (tp * pp)  # 16 / (1*4) = 4
    mbs = 2
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
            if batch_size % (dp * mbs) != 0:
                # Final fallback
                tp, pp, dp, mbs, ac = 2, 4, 2, 2, False
    
    # ---- General fallback logic for other workloads ----
    if total_gpus != 16 or gpu_memory_gb != 40:
        # General strategy based on learnings:
        # Higher PP is very beneficial (pp=4 >> pp=2 >> pp=1)
        # Lower tp preferred if memory allows (reduces communication)
        # Balance dp for good accumulation step count
        
        # Start with tp=1 or tp=2 depending on model size
        num_layers = workload.get("num_layers", 32)
        num_heads = workload.get("num_heads", 32)
        
        # Estimate model params roughly
        hidden_dim = num_heads * 128  # rough estimate
        est_params_b = num_layers * 12 * (hidden_dim ** 2) / 1e9
        
        # Try to maximize pp first, then dp, minimize tp
        best_config = None
        best_heuristic = float('inf')
        
        for tp in tp_choices:
            if tp > gpus_per_node:
                continue
            for pp in sorted(pp_choices, reverse=True):
                dp = total_gpus // (tp * pp)
                if dp not in dp_choices or dp < 1 or tp * pp * dp != total_gpus:
                    continue
                
                # Memory check: params per stage per tp shard
                params_per_gpu_gb = (est_params_b * 2) / (pp * tp)  # bf16
                optimizer_per_gpu_gb = params_per_gpu_gb * 4  # Adam states
                total_mem = params_per_gpu_gb + optimizer_per_gpu_gb
                
                if total_mem > gpu_memory_gb * 0.7 and not ac:
                    continue
                
                for mbs in sorted(mbs_choices, reverse=True):
                    if batch_size % (dp * mbs) != 0:
                        continue
                    acc_steps = batch_size // (dp * mbs)
                    if acc_steps < pp:  # Need enough micro-batches for pipeline
                        continue
                    
                    # Heuristic: prefer high pp, low tp, moderate dp
                    # Lower score = better
                    h = tp * 2.0 + (1.0 / pp) * 10.0 + acc_steps * 0.5
                    
                    if h < best_heuristic:
                        best_heuristic = h
                        best_config = (tp, pp, dp, mbs, False)
        
        if best_config:
            tp, pp, dp, mbs, ac = best_config
        else:
            tp, pp, dp, mbs, ac = 2, 4, 2, 2, False
    
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "micro_batch_size": mbs,
        "activation_checkpointing": ac,
    }
