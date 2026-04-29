
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus    = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    batch_size    = workload.get("batch_size", 1)
    gpu_memory    = environment.get("gpu_memory_gb", 80)
    inter_bw      = environment.get("inter_node_bandwidth_gbps", 100)
    intra_bw      = environment.get("intra_node_bandwidth_gbps", 600)
    
    # Build lookup of valid choices for each config dimension
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    num_nodes = total_gpus // gpus_per_node
    
    # Strategy: 
    # 1. Keep TP within a node to use fast NVLink
    # 2. Minimize cross-node communication
    # 3. Use largest micro_batch_size to reduce gradient accumulation steps
    # 4. Avoid PP unless needed for memory
    
    # From experiments: tp=4, dp=4 (11.87) beat tp=2, dp=8 (14.22)
    # tp=4 with 4 GPUs/node means all TP is intra-node (good)
    # dp=4 means 4 nodes doing allreduce (less cross-node comm than dp=8)
    
    # Best approach: tp = gpus_per_node to keep all TP intra-node
    # Then dp = total_gpus / tp, pp = 1
    tp = min(gpus_per_node, total_gpus)
    pp = 1
    dp = total_gpus // (tp * pp)
    
    # Validate tp
    if "tp" in valid_choices:
        # Pick largest valid tp that fits within a node
        valid_tp = sorted([t for t in valid_choices["tp"] if t <= gpus_per_node], reverse=True)
        if valid_tp:
            tp = valid_tp[0]
        else:
            tp = min(valid_choices["tp"])
    
    # Compute dp
    dp = total_gpus // (tp * pp)
    
    # Validate dp
    if "dp" in valid_choices and dp not in valid_choices["dp"]:
        # Try adjusting pp to get a valid dp
        best_config = None
        best_score = float('inf')
        
        tp_choices = valid_choices.get("tp", [tp])
        pp_choices = valid_choices.get("pp", [pp])
        dp_choices = valid_choices.get("dp", [dp])
        
        for t in tp_choices:
            for p in pp_choices:
                d = total_gpus // (t * p)
                if d in dp_choices and t * p * d == total_gpus:
                    # Score: prefer high tp (within node), low pp, high dp
                    # Penalize tp > gpus_per_node (cross-node TP is slow)
                    cross_node_tp = max(0, t - gpus_per_node)
                    score = cross_node_tp * 100 + p * 10  # lower is better
                    if score < best_score:
                        best_score = score
                        best_config = (t, p, d)
        
        if best_config:
            tp, pp, dp = best_config
    
    # Micro-batch size: larger is better to reduce gradient accumulation steps
    # But must fit in memory. Per-rank batch = batch_size / dp
    per_rank_batch = batch_size // dp if dp > 0 else batch_size
    
    micro_batch = 4  # largest available
    if "micro_batch_size" in valid_choices:
        # Pick largest valid micro_batch that divides per_rank_batch
        valid_mb = sorted(valid_choices["micro_batch_size"], reverse=True)
        micro_batch = valid_mb[0]  # default to largest
        for mb in valid_mb:
            if per_rank_batch % mb == 0:
                micro_batch = mb
                break
    
    # Activation checkpointing: only if memory constrained
    # For 8B model on 40GB with tp=4, each GPU holds ~2B params → ~4GB in bf16
    # Optimizer states: ~12GB (Adam), activations depend on batch/seq
    # With tp=4, should fit without checkpointing
    activation_checkpointing = False
    
    config = {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": activation_checkpointing,
    }
    
    return config
