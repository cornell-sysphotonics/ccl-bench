
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
    
    # Strategy learned from experiments:
    # - tp=4 within node (NVLink) >> tp=2 with more dp across nodes
    # - pp=1 avoids pipeline bubble overhead
    # - micro_batch=2 slightly better than micro_batch=4 for this workload
    # Now try micro_batch=1 to see if trend continues
    
    # Enumerate all valid (tp, pp, dp) combos where tp*pp*dp == total_gpus
    tp_choices = valid_choices.get("tp", [1])
    pp_choices = valid_choices.get("pp", [1])
    dp_choices = valid_choices.get("dp", [1])
    
    best_config = None
    best_score = float('inf')
    
    for t in tp_choices:
        for p in pp_choices:
            d = total_gpus // (t * p)
            if d in dp_choices and t * p * d == total_gpus:
                # Scoring heuristic (lower is better):
                # 1. Strongly prefer TP within a single node (intra-node NVLink)
                cross_node_tp = t > gpus_per_node
                # 2. Minimize dp to reduce cross-node allreduce
                # 3. Avoid pp (pipeline bubbles)
                # 4. TP within node is great - prefer max TP within node
                
                score = 0
                if cross_node_tp:
                    score += 1000  # heavy penalty
                score += d * 10   # penalize high dp (more allreduce)
                score += p * 50   # penalize pp (pipeline bubbles)
                score -= t * 5    # reward higher tp within node
                
                if score < best_score:
                    best_score = score
                    best_config = (t, p, d)
    
    if best_config:
        tp, pp, dp = best_config
    else:
        tp, pp, dp = 4, 1, 4  # fallback
    
    # Micro-batch size: try 1 this time (experiments show 2 > 4, try 1)
    per_rank_batch = batch_size // dp if dp > 0 else batch_size
    
    micro_batch = 1  # trying smallest to see if trend continues
    if "micro_batch_size" in valid_choices:
        if micro_batch not in valid_choices["micro_batch_size"]:
            micro_batch = min(valid_choices["micro_batch_size"])
    
    # Activation checkpointing: off for speed (model fits in memory with tp=4)
    activation_checkpointing = False
    
    config = {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": activation_checkpointing,
    }
    
    return config
