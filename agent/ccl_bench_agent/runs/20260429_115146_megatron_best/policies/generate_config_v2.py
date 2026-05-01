
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Optimized config for LLM training on multi-node GPU clusters.
    
    Lessons learned:
    - tp=4, dp=4, pp=1, mbs=2, ac=False -> 8.358 (success)
    - tp=4, dp=4, pp=1, mbs=4, ac=False -> SIGSEGV (OOM likely)
    
    Strategy for this iteration:
    - Try tp=2 to reduce TP communication overhead (tp=4 uses all intra-node GPUs for TP)
    - With tp=2, dp=8 maximizes data parallelism across all 16 GPUs
    - mbs=2 to balance memory and computation efficiency
    - ac=False to avoid recompute overhead (model fits in memory with tp=2)
    """
    total_gpus = environment.get("total_gpus", 16)
    gpus_per_node = environment.get("gpus_per_node", 4)
    gpu_memory_gb = environment.get("gpu_memory_gb", 40)
    
    # Extract config space to know valid choices
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
    
    # Estimate model size in billions of parameters
    model_size_b = 8.0  # default
    if "8b" in model_family:
        model_size_b = 8.0
    elif "70b" in model_family:
        model_size_b = 70.0
    elif "13b" in model_family:
        model_size_b = 13.0
    elif "7b" in model_family:
        model_size_b = 7.0
    
    # Bytes per parameter: bf16 = 2 bytes for weights, ~10 bytes total with optimizer
    bytes_per_param = 10 if precision == "bf16" else 12
    model_memory_gb = model_size_b * bytes_per_param  # Total memory needed
    
    best_config = None
    best_score = float('inf')
    
    for tp in tp_choices:
        # Keep TP within a node for fast communication
        if tp > gpus_per_node:
            continue
        
        for pp in pp_choices:
            dp = total_gpus // (tp * pp)
            
            if dp not in dp_choices:
                continue
            if dp < 1:
                continue
            if tp * pp * dp != total_gpus:
                continue
            
            # Memory check: model memory per GPU after TP and PP splitting
            mem_per_gpu = model_memory_gb / (tp * pp)
            
            for mbs in mbs_choices:
                # Check batch size divisibility
                if batch_size % (dp * mbs) != 0:
                    continue
                acc_steps = batch_size // (dp * mbs)
                if acc_steps < 1:
                    continue
                
                # Activation memory estimate (rough): proportional to mbs * seq_len / tp
                act_mem = mbs * seq_len * model_size_b * 0.001 / tp  # rough GB estimate
                
                for ac in ac_choices:
                    act_mem_effective = act_mem * 0.3 if ac else act_mem
                    total_mem = mem_per_gpu + act_mem_effective
                    
                    # Skip if likely OOM (leave some headroom)
                    if total_mem > gpu_memory_gb * 0.85 and not ac:
                        continue
                    
                    # Heuristic scoring (lower is better)
                    score = 0.0
                    
                    # TP communication cost
                    # Each TP step requires 2 all-reduces per layer
                    # Cost depends on message size and bandwidth
                    if tp > 1:
                        # TP comm is proportional to hidden_size / tp * 2 * num_layers
                        # Within node: use intra_bw, fast
                        tp_comm_cost = 0.15 * (tp - 1)  # Roughly linear in TP degree
                    else:
                        tp_comm_cost = 0.0
                    
                    # PP bubble cost: (pp-1)/pp fraction of time wasted in bubbles
                    if pp > 1:
                        pp_bubble_cost = 3.0 * (pp - 1) / pp  # Significant overhead
                    else:
                        pp_bubble_cost = 0.0
                    
                    # DP all-reduce cost
                    # With more DP, the all-reduce is over more GPUs (potentially cross-node)
                    # But the gradient size per GPU is smaller with TP
                    if dp > 1:
                        num_nodes_dp = max(1, dp * tp * pp // (gpus_per_node * tp * pp))
                        # Actually: nodes = total_gpus / gpus_per_node
                        # Cross-node DP adds latency
                        cross_node = dp > (gpus_per_node // tp)
                        if cross_node:
                            dp_cost = 0.3 + 0.05 * dp  # Cross-node penalty
                        else:
                            dp_cost = 0.05 * dp  # Intra-node only
                    else:
                        dp_cost = 0.0
                    
                    # Compute cost per step: base compute / (tp * pp), times acc_steps
                    # Larger mbs = better GPU utilization (less kernel launch overhead)
                    compute_per_microbatch = 1.0 / (tp * pp) * (1.0 / (mbs ** 0.2))
                    total_compute = compute_per_microbatch * acc_steps
                    
                    # Activation checkpointing overhead: ~33% more compute
                    if ac:
                        total_compute *= 1.33
                    
                    total_score = tp_comm_cost * acc_steps + pp_bubble_cost + dp_cost + total_compute
                    
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
            "micro_batch_size": 2,
            "activation_checkpointing": True,
        }
    
    return best_config
