
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
    
    # Lessons learned from history:
    # 1. dp=1 is FAR better than dp=2 (0.47 vs 5.76) - inter-node allreduce is very costly
    # 2. tp=8, pp=1, dp=1, mbs=1 = 0.4743 (BEST)
    # 3. tp=8, pp=1, dp=1, mbs=2 = 0.4866
    # 4. tp=4, pp=2, dp=1 = 0.5722
    # 5. mbs=4 with tp=4 caused OOM, but tp=8 splits model more
    
    # Strategy: strongly prefer dp=1 to avoid inter-node allreduce
    # Then prefer configurations that minimize total communication
    
    best_score = float('inf')
    best_config = None
    
    for tp in sorted(valid_tps, reverse=True):
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
                
                for ac in valid_ac:
                    # Memory check - rough estimate
                    # tp=8 splits model 8 ways, tp=4 splits 4 ways
                    # mbs=4 with tp=4 caused OOM on 40GB A100
                    model_mem_per_gpu = 16.0 / tp  # ~16GB for 8B model in bf16
                    act_mem = mbs * 1.0 * (1.0 if not ac else 0.3) / tp
                    total_mem_est = model_mem_per_gpu + act_mem * 3  # rough
                    
                    if total_mem_est > gpu_memory_gb * 0.85 and not ac:
                        continue
                    
                    # Cost model based on observed data
                    cost = 0.0
                    
                    # DP cost: inter-node allreduce is extremely expensive
                    if dp > 1:
                        # dp=2 with tp=4,pp=1 gave ~5.76-5.89 vs dp=1 giving ~0.47
                        # This is a ~12x penalty!
                        dp_crosses_nodes = num_nodes > 1
                        if dp_crosses_nodes:
                            cost += dp * 5.0  # Massive penalty
                        else:
                            cost += dp * 0.1
                    
                    # TP communication cost
                    tp_crosses_nodes = tp > gpus_per_node
                    if tp_crosses_nodes:
                        # tp=8 across 2 nodes still gave 0.47, so not that bad
                        # Megatron handles this with NVLink-like within node + slower cross-node
                        cost += (tp - gpus_per_node) * 0.02 + (gpus_per_node - 1) * 0.005
                    else:
                        cost += (tp - 1) * 0.005
                    
                    # PP cost: pipeline bubble
                    if pp > 1:
                        bubble_frac = (pp - 1) / (num_microbatches + pp - 1)
                        cost += bubble_frac * 0.5
                        # PP p2p communication
                        pp_crosses_nodes = tp * pp > gpus_per_node
                        if pp_crosses_nodes:
                            cost += (pp - 1) * 0.03
                        else:
                            cost += (pp - 1) * 0.005
                    
                    # Compute: larger mbs slightly better utilization
                    # But mbs=1 gave 0.4743, mbs=2 gave 0.4866 for tp=8
                    # So mbs=1 is actually better (less memory pressure, faster)
                    compute_cost = 0.45 + 0.01 * mbs
                    cost += compute_cost
                    
                    # Activation checkpointing adds recompute overhead
                    if ac:
                        cost += 0.1
                    
                    if cost < best_score:
                        best_score = cost
                        best_config = {
                            "tp": tp,
                            "dp": dp,
                            "pp": pp,
                            "micro_batch_size": mbs,
                            "activation_checkpointing": ac,
                        }
    
    if best_config is None:
        # Fallback to known best config
        best_config = {
            "tp": 8,
            "dp": 1,
            "pp": 1,
            "micro_batch_size": 1,
            "activation_checkpointing": False,
        }
    
    return best_config
