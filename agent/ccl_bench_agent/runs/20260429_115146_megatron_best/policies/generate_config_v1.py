
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Optimized config for LLM training on multi-node GPU clusters.
    
    Key principles:
    1. Keep tensor parallelism (TP) within a node to leverage high intra-node bandwidth
    2. Minimize pipeline parallelism to avoid bubble overhead
    3. Maximize data parallelism for throughput
    4. Choose micro_batch_size to minimize gradient accumulation steps
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
    
    batch_size = workload.get("batch_size", 32)
    
    # Model size heuristic based on model family
    model_family = workload.get("model_family", "").lower()
    
    # Estimate model size for memory planning
    # Llama-3.1-8B ~ 8B params, ~16GB in bf16, ~32GB with optimizer states
    # With TP=2, each GPU holds ~8GB model + ~16GB optimizer = ~24GB, fits in 40GB
    
    # Strategy: Keep TP intra-node, minimize PP, maximize DP
    # TP=2 is a good sweet spot - reduces memory per GPU while keeping comm fast
    
    tp_choices = sorted(valid_choices.get("tp", [1, 2, 4, 8]))
    pp_choices = sorted(valid_choices.get("pp", [1, 2, 4, 8]))
    dp_choices = sorted(valid_choices.get("dp", [1, 2, 4, 8, 16]))
    mbs_choices = sorted(valid_choices.get("micro_batch_size", [1, 2, 4]))
    ac_choices = valid_choices.get("activation_checkpointing", [True, False])
    
    best_config = None
    best_score = float('inf')
    
    for tp in tp_choices:
        # Keep TP within a node
        if tp > gpus_per_node:
            continue
        
        for pp in pp_choices:
            dp = total_gpus // (tp * pp)
            
            # dp must be valid
            if dp not in dp_choices:
                continue
            if dp < 1:
                continue
            if tp * pp * dp != total_gpus:
                continue
            
            for mbs in mbs_choices:
                # Global batch = dp * mbs * num_accumulation_steps
                # We need dp * mbs * acc = batch_size
                if batch_size % (dp * mbs) != 0:
                    continue
                acc_steps = batch_size // (dp * mbs)
                if acc_steps < 1:
                    continue
                
                for ac in ac_choices:
                    # Score this config with heuristics
                    score = 0.0
                    
                    # TP communication cost - scales with model size and inversely with bandwidth
                    if tp > 1:
                        # Intra-node TP is fast, inter-node TP is slow
                        nodes_in_tp = max(1, tp // gpus_per_node)
                        if nodes_in_tp > 1:
                            tp_cost = tp * 2.0  # Heavy penalty for cross-node TP
                        else:
                            tp_cost = tp * 0.3  # Intra-node TP is relatively cheap
                    else:
                        tp_cost = 0.0
                    
                    # PP bubble overhead
                    if pp > 1:
                        pp_cost = (pp - 1) / pp * 2.0  # Pipeline bubble fraction
                    else:
                        pp_cost = 0.0
                    
                    # DP all-reduce cost - depends on model size and inter-node comm
                    if dp > 1:
                        # Some DP ranks are cross-node
                        dp_cost = 0.5 * (dp / total_gpus)  # Normalized
                    else:
                        dp_cost = 0.0
                    
                    # Gradient accumulation overhead
                    acc_cost = acc_steps * 0.3  # Each acc step adds overhead
                    
                    # Computation time (inversely proportional to parallelism, but with overhead)
                    compute_cost = 5.0 / (tp * pp)  # Base compute per GPU
                    
                    # Memory pressure - activation checkpointing slows things down
                    if ac:
                        compute_cost *= 1.3  # ~30% recompute overhead
                    
                    # Micro-batch size efficiency - larger is generally better for GPU utilization
                    mbs_efficiency = 1.0 / (mbs ** 0.3)
                    
                    total_score = tp_cost + pp_cost + dp_cost + acc_cost + compute_cost * mbs_efficiency
                    
                    if total_score < best_score:
                        best_score = total_score
                        best_config = {
                            "tp": tp,
                            "pp": pp,
                            "dp": dp,
                            "micro_batch_size": mbs,
                            "activation_checkpointing": ac,
                        }
    
    # Fallback
    if best_config is None:
        best_config = {
            "tp": 2,
            "pp": 1,
            "dp": 8,
            "micro_batch_size": 4,
            "activation_checkpointing": False,
        }
    
    return best_config
