
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Iteration 15 -> 16: Stick with best known config (tp=4, pp=4, dp=1, mbs=1, ac=False = 0.4427).
    
    Key learnings from 15 iterations:
    - dp=1 is strongly preferred (minimizes all-reduce communication)
    - tp=4 (within node, gpus_per_node=4) is optimal
    - pp=4 to use remaining GPUs
    - mbs=1 slightly beats mbs=2 (lower bubble ratio: 9.375% vs 18.75%)
    - ac=False is better (no recompute overhead)
    
    Attempting to reproduce/confirm best score.
    General heuristic updated with all learnings.
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
    
    # General optimization: find best config using learned heuristic
    best_config = None
    best_heuristic = float('inf')
    
    for tp_c in tp_choices:
        # TP must stay within a node for good performance
        if tp_c > gpus_per_node:
            continue
        
        for pp_c in pp_choices:
            dp_c = total_gpus // (tp_c * pp_c)
            if dp_c not in dp_choices or dp_c < 1 or tp_c * pp_c * dp_c != total_gpus:
                continue
            
            # Memory estimation (rough)
            # Model params per GPU in GB
            params_per_gpu_gb = (est_params_b * bytes_per_param) / (pp_c * tp_c)
            # Optimizer states (~12 bytes per param for Adam with bf16 mixed precision)
            optimizer_per_gpu_gb = (est_params_b * 12) / (pp_c * tp_c * 1e9) * 1e9 / 1e9
            # Activation memory (rough)
            act_per_gpu_gb = (batch_size / max(dp_c, 1)) * seq_len * hidden_dim * bytes_per_param * (num_layers / pp_c) / (tp_c * 1e9)
            
            total_mem = params_per_gpu_gb + optimizer_per_gpu_gb
            
            # Skip if clearly OOM (leaving room for activations)
            if total_mem > gpu_memory_gb * 0.8:
                continue
            
            for mbs_c in mbs_choices:
                if batch_size % (dp_c * mbs_c) != 0:
                    continue
                acc_steps_c = batch_size // (dp_c * mbs_c)
                if acc_steps_c < 1:
                    continue
                # Pipeline needs at least pp stages worth of micro-batches
                if pp_c > 1 and acc_steps_c < pp_c:
                    continue
                
                # Heuristic cost model learned from experiments:
                # 
                # Key finding: dp cost dominates everything.
                # dp=1 >> dp=2 >> dp=4 in terms of performance.
                # Within dp=1: tp=gpus_per_node, pp=total_gpus/tp is best.
                # mbs=1 (lowest bubble) is best when dp=1.
                
                h = 0.0
                
                # Data parallelism cost (all-reduce over inter-node links)
                # This is the DOMINANT factor. Each additional dp rank adds
                # massive communication overhead, especially inter-node.
                num_nodes = total_gpus // gpus_per_node
                # dp across nodes is very expensive
                dp_across_nodes = max(1, dp_c // (total_gpus // (tp_c * gpus_per_node * pp_c) if tp_c * pp_c <= gpus_per_node else 1))
                h += dp_c * 100.0  # Heavily penalize dp > 1
                
                # Pipeline bubble cost
                if pp_c > 1 and acc_steps_c > 0:
                    bubble_frac = (pp_c - 1) / acc_steps_c
                    h += bubble_frac * 20.0
                
                # TP cost (within node is cheap)
                if tp_c <= gpus_per_node:
                    h += tp_c * 1.0  # Small cost for TP communication
                else:
                    h += tp_c * 50.0  # Inter-node TP is terrible
                
                # Prefer tp = gpus_per_node (use full node bandwidth)
                h += abs(tp_c - gpus_per_node) * 2.0
                
                # mbs cost: smaller mbs = more acc steps = less bubble
                # But also more kernel launches. For pp>1, smaller mbs is better.
                if pp_c > 1:
                    h += mbs_c * 1.0
                else:
                    # For pp=1, larger mbs is slightly better (fewer steps)
                    h += (1.0 / mbs_c) * 0.5
                
                if h < best_heuristic:
                    best_heuristic = h
                    best_config = (tp_c, pp_c, dp_c, mbs_c, False)
    
    if best_config:
        tp, pp, dp, mbs, ac = best_config
    else:
        # Fallback: simple config
        tp = min(gpus_per_node, max(tp_choices))
        pp = 1
        dp = total_gpus // tp
        if dp not in dp_choices:
            dp = max(d for d in dp_choices if d <= total_gpus // tp)
        mbs = mbs_choices[0]
        ac = False
    
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "micro_batch_size": mbs,
        "activation_checkpointing": ac,
    }
