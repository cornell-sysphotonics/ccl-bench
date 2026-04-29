
def generate_config(workload: dict, environment: dict) -> dict:
    total_gpus = environment.get("total_gpus", 8)
    gpus_per_node = environment.get("gpus_per_node", 4)
    batch_size = workload.get("batch_size", 8)
    gpu_memory_gb = environment.get("gpu_memory_gb", 40)
    inter_bw = environment.get("inter_node_bandwidth_gbps", 25)
    intra_bw = environment.get("intra_node_bandwidth_gbps", 600)
    num_nodes = total_gpus // gpus_per_node
    
    # Build valid choices lookup
    config_space = workload.get("config_space", [])
    choices = {}
    for dim in config_space:
        choices[dim["key"]] = dim["choices"]
    
    valid_tps = choices.get("tp", [1, 2, 4, 8])
    valid_pps = choices.get("pp", [1, 2, 4])
    valid_dps = choices.get("dp", [1, 2, 4, 8])
    valid_mbs = choices.get("micro_batch_size", [1, 2, 4])
    valid_ac = choices.get("activation_checkpointing", [True, False])
    
    # Strategy based on lessons learned:
    # 1. Keep TP within a single node (tp <= gpus_per_node) for fast intra-node comm
    # 2. Use PP to span across nodes (avoids expensive allreduce, only point-to-point)
    # 3. Minimize dp to avoid inter-node allreduce
    # 4. Choose mbs to balance pipeline bubble vs per-microbatch overhead
    
    best_score = float('inf')
    best_config = None
    
    for tp in sorted(valid_tps, reverse=True):
        if tp > gpus_per_node:
            continue  # Don't let TP cross node boundaries
        if tp > total_gpus:
            continue
            
        for pp in valid_pps:
            if tp * pp > total_gpus:
                continue
            dp = total_gpus // (tp * pp)
            if dp not in valid_dps:
                continue
            if dp * tp * pp != total_gpus:
                continue
            
            for mbs in valid_mbs:
                per_dp_batch = batch_size // dp
                if per_dp_batch <= 0:
                    continue
                if per_dp_batch % mbs != 0:
                    continue
                num_microbatches = per_dp_batch // mbs
                
                # Estimate cost heuristics
                
                # Pipeline bubble fraction
                if pp > 1:
                    bubble_frac = (pp - 1) / (num_microbatches + pp - 1)
                else:
                    bubble_frac = 0
                
                # TP communication cost (allreduce within TP group)
                # Higher TP = more communication but within node = fast
                tp_comm_cost = (tp - 1) * 0.01 if tp <= gpus_per_node else (tp - 1) * 0.5
                
                # PP communication cost (point-to-point between stages)
                # If PP crosses nodes, slower but still just p2p
                pp_crosses_nodes = pp > 1 and (tp * pp > gpus_per_node)
                pp_comm_cost = 0
                if pp > 1:
                    if pp_crosses_nodes:
                        pp_comm_cost = (pp - 1) * 0.05  # Inter-node p2p is manageable
                    else:
                        pp_comm_cost = (pp - 1) * 0.01
                
                # DP communication cost (allreduce across DP ranks)
                # This is the most expensive if it crosses nodes
                dp_comm_cost = 0
                if dp > 1:
                    dp_crosses_nodes = (dp * tp > gpus_per_node) or (num_nodes > 1 and dp > 1)
                    if dp_crosses_nodes:
                        dp_comm_cost = dp * 1.0  # Very expensive - inter-node allreduce
                    else:
                        dp_comm_cost = dp * 0.05
                
                # Compute efficiency: larger mbs = better GPU utilization but more memory
                compute_eff = 1.0 / (1.0 + 0.1 / mbs)  # diminishing returns
                
                # Memory pressure: larger mbs and no checkpointing uses more memory
                mem_pressure = mbs * (1.0 if tp <= 2 else 0.5)
                if mem_pressure > 4 and gpu_memory_gb <= 40:
                    continue  # Skip likely OOM configs
                
                # Overall estimated cost
                base_compute = 1.0 / compute_eff
                total_cost = base_compute * (1 + bubble_frac) + tp_comm_cost + pp_comm_cost + dp_comm_cost
                
                if total_cost < best_score:
                    best_score = total_cost
                    best_config = {
                        "tp": tp,
                        "dp": dp,
                        "pp": pp,
                        "micro_batch_size": mbs,
                        "activation_checkpointing": False,
                    }
    
    if best_config is None:
        # Fallback to known best
        best_config = {
            "tp": min(gpus_per_node, max(valid_tps)),
            "dp": 1,
            "pp": 2 if 2 in valid_pps else 1,
            "micro_batch_size": 2 if 2 in valid_mbs else 1,
            "activation_checkpointing": False,
        }
    
    # Ensure activation_checkpointing is valid
    if best_config["activation_checkpointing"] not in valid_ac:
        best_config["activation_checkpointing"] = valid_ac[0]
    
    return best_config
