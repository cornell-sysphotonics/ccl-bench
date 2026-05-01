
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Iteration 16 -> 17: Try tp=4, pp=2, dp=2, mbs=1, ac=False.
    
    Best known: tp=4, pp=4, dp=1, mbs=1, ac=False = 0.4427
    
    Now trying dp=2, pp=2 to see if lower pipeline bubble compensates for dp cost.
    With pp=2, dp=2, mbs=1: acc_steps = 32/(2*1) = 16, bubble = 1/16 = 6.25%
    vs pp=4, dp=1, mbs=1: acc_steps = 32/(1*1) = 32, bubble = 3/32 = 9.375%
    
    The question is whether reduced bubble (6.25% vs 9.375%) can offset dp=2 all-reduce cost.
    
    History summary (dp=1 configs):
    - tp=4, pp=4, dp=1, mbs=1, ac=False → 0.4427 (best)
    - tp=4, pp=4, dp=1, mbs=2, ac=False → 0.4707
    - tp=8, pp=2, dp=1, mbs=1, ac=False → 0.6068
    - tp=2, pp=8, dp=1, mbs=1, ac=False → 1.031
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
    num_layers = workload.get("num_layers", 32)
    num_heads = workload.get("num_heads", 32)
    
    tp_choices = sorted(valid_choices.get("tp", [1, 2, 4, 8]))
    pp_choices = sorted(valid_choices.get("pp", [1, 2, 4, 8]))
    dp_choices = sorted(valid_choices.get("dp", [1, 2, 4, 8, 16]))
    mbs_choices = sorted(valid_choices.get("micro_batch_size", [1, 2, 4]))
    ac_choices = valid_choices.get("activation_checkpointing", [True, False])
    
    # Estimate model parameters
    hidden_dim = num_heads * 128  # head_dim=128 for most modern models
    est_params_b = num_layers * 12 * (hidden_dim ** 2) / 1e9
    bytes_per_param = 2 if precision in ["bf16", "fp16"] else 4
    
    # Try the unexplored config: tp=4, pp=2, dp=2, mbs=1
    # If this doesn't beat 0.4427, we'll go back to best known
    
    # General optimization logic
    best_config = None
    best_heuristic = float('inf')
    
    for tp_c in tp_choices:
        if tp_c > gpus_per_node:
            continue
        
        for pp_c in pp_choices:
            dp_c = total_gpus // (tp_c * pp_c)
            if dp_c not in dp_choices or dp_c < 1 or tp_c * pp_c * dp_c != total_gpus:
                continue
            
            # Memory check
            params_per_gpu_gb = (est_params_b * bytes_per_param) / (pp_c * tp_c)
            optimizer_per_gpu_gb = (est_params_b * 12) / (pp_c * tp_c)
            total_mem = params_per_gpu_gb + optimizer_per_gpu_gb
            if total_mem > gpu_memory_gb * 0.8:
                continue
            
            for mbs_c in mbs_choices:
                if batch_size % (dp_c * mbs_c) != 0:
                    continue
                acc_steps_c = batch_size // (dp_c * mbs_c)
                if acc_steps_c < 1:
                    continue
                if pp_c > 1 and acc_steps_c < pp_c:
                    continue
                
                # Refined heuristic based on all 16 runs
                # Calibrated costs:
                #   dp=1 → 0.4427 (tp=4,pp=4,mbs=1)
                #   dp=2 → 0.6739 (tp=4,pp=2,mbs=2) best dp=2
                #   dp=4 → 1.332 (tp=4,pp=1,mbs=2) best dp=4
                
                h = 0.0
                
                # DP cost: dominant factor, especially inter-node
                # Empirically: dp=1 saves ~0.23 over dp=2 best case
                if dp_c > 1:
                    # Check if dp is within node or across nodes
                    nodes_for_tp_pp = max(1, (tp_c * pp_c) // gpus_per_node)
                    # dp communication is all-reduce, scales with model size
                    h += dp_c * 0.15  # calibrated from data
                
                # Pipeline bubble cost
                if pp_c > 1:
                    bubble_frac = (pp_c - 1) / acc_steps_c
                    h += bubble_frac * 0.5
                
                # TP cost within node
                if tp_c <= gpus_per_node:
                    h += tp_c * 0.02
                else:
                    h += tp_c * 0.5  # inter-node TP very expensive
                
                # Prefer tp = gpus_per_node
                h += abs(tp_c - gpus_per_node) * 0.05
                
                # PP inter-node communication (point-to-point, cheaper than all-reduce)
                pp_nodes = pp_c  # rough: each pp stage on different node
                if pp_c > 1:
                    h += pp_c * 0.01  # small cost per pp stage
                
                # MBS effect: smaller is better for pp>1 (less bubble)
                # Already captured in bubble_frac
                
                if h < best_heuristic:
                    best_heuristic = h
                    best_config = (tp_c, pp_c, dp_c, mbs_c, False)
    
    if best_config:
        tp, pp, dp, mbs, ac = best_config
    else:
        tp = min(gpus_per_node, max(tp_choices))
        pp = total_gpus // tp
        dp = 1
        mbs = mbs_choices[0]
        ac = False
    
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "micro_batch_size": mbs,
        "activation_checkpointing": ac,
    }
