
def policy(
    model: str,
    batch_size: int,
    seq_len: int,
    dmodel: int,
    num_heads: int,
    num_kv_heads: int,
    num_stacks: int,
    precision: str,
    total_gpus: int,
    gpu_memory_gb: int,
    gpus_per_node: int,
    intra_node_bandwidth_gbps: int,
    inter_node_bandwidth_gbps: int
) -> dict:
    """Return {"tp": int, "dp": int, "pp": int, "micro_batch": int} for the given workload.
    
    Strategy:
    - Estimate model memory and activation memory to determine minimum TP and PP
    - Use DP to maximize throughput
    - Keep micro_batch as small as possible (= batch_size // dp) to reduce memory
    """
    import math
    
    # Bytes per parameter based on precision
    bytes_per_param = 4 if precision == "fp32" else 2
    
    # Estimate model parameters (rough for transformer models)
    # For a transformer: ~12 * dmodel^2 * num_stacks parameters (approximate)
    params = 12 * dmodel * dmodel * num_stacks
    model_memory_bytes = params * bytes_per_param
    
    # Optimizer states: ~3x model for Adam (params + momentum + variance) in fp32
    optimizer_memory_bytes = params * 4 * 3  # Always fp32 for optimizer states
    
    # Activation memory per sample per layer (rough estimate)
    # ~= seq_len * dmodel * batch_size * bytes_per_param * some_factor
    # For transformers: roughly 34 * hidden * seq * micro_batch * bytes (per layer)
    
    # Valid TP values must divide both num_heads and num_kv_heads
    valid_tps = []
    for t in [1, 2, 4, 8, 16, 32]:
        if t <= total_gpus and num_heads % t == 0 and num_kv_heads % t == 0:
            valid_tps.append(t)
    
    # Valid PP values
    valid_pps = [1, 2, 4, 8, 16, 32]
    
    best_config = None
    best_score = float('inf')
    
    for tp in valid_tps:
        for pp in valid_pps:
            if tp * pp > total_gpus:
                continue
            
            dp = total_gpus // (tp * pp)
            if dp < 1:
                continue
            
            # micro_batch must be >= batch_size / dp (ceiling)
            min_mb = math.ceil(batch_size / dp)
            if min_mb < 1:
                min_mb = 1
            
            # Try a few micro_batch sizes
            mb_candidates = [min_mb]
            # Also try some larger values
            for mult in [2, 4, 8]:
                if min_mb * mult <= batch_size:
                    mb_candidates.append(min_mb * mult)
            
            for micro_batch in mb_candidates:
                # Estimate memory per GPU
                # Model memory split across TP and PP
                layers_per_pp = math.ceil(num_stacks / pp)
                model_mem_per_gpu = (12 * dmodel * dmodel * layers_per_pp * bytes_per_param) / tp
                
                # Optimizer memory (split same way, but always fp32)
                opt_mem_per_gpu = (12 * dmodel * dmodel * layers_per_pp * 4 * 3) / tp
                
                # Activation memory per micro-batch per layer
                # Rough: seq_len * dmodel * micro_batch * bytes * factor / tp
                # Factor accounts for attention, FFN intermediates, etc.
                activation_factor = 34  # rough constant for transformer activations
                act_mem_per_gpu = (seq_len * dmodel * micro_batch * bytes_per_param * activation_factor * layers_per_pp) / tp
                
                total_mem = model_mem_per_gpu + opt_mem_per_gpu + act_mem_per_gpu
                total_mem_gb = total_mem / (1024**3)
                
                # Add some safety margin
                if total_mem_gb > gpu_memory_gb * 0.85:
                    continue
                
                # Estimate wall time (heuristic scoring)
                # Compute time scales with: micro_batch * seq_len * params_per_gpu * num_microbatches
                num_microbatches = math.ceil(batch_size / (dp * micro_batch))
                params_per_gpu = 12 * dmodel * dmodel * layers_per_pp / tp
                
                compute_time = micro_batch * seq_len * params_per_gpu * num_microbatches
                
                # Communication costs
                # TP all-reduce: proportional to dmodel^2 * seq_len * micro_batch / bandwidth
                # Within node: use intra_node_bandwidth, across nodes: inter_node_bandwidth
                tp_crosses_nodes = tp > gpus_per_node
                tp_bw = inter_node_bandwidth_gbps if tp_crosses_nodes else intra_node_bandwidth_gbps
                tp_comm = (dmodel * dmodel * seq_len * micro_batch * bytes_per_param * num_microbatches * 2) / (tp_bw * 1e9 / 8) if tp > 1 else 0
                
                # DP all-reduce: proportional to model_size / bandwidth
                dp_crosses_nodes = dp > 1 and (tp * pp) < total_gpus // gpus_per_node * gpus_per_node
                dp_bw = inter_node_bandwidth_gbps if dp > gpus_per_node else intra_node_bandwidth_gbps
                # Actually, if we have multiple nodes, DP likely crosses nodes
                num_nodes_used = total_gpus // gpus_per_node
                if dp > 1 and num_nodes_used > 1:
                    dp_bw = inter_node_bandwidth_gbps
                dp_comm = (params_per_gpu * bytes_per_param * 2 * (dp - 1) / dp) / (dp_bw * 1e9 / 8) if dp > 1 else 0
                
                # PP communication: send/recv activations between stages
                pp_crosses_nodes = pp > gpus_per_node
                pp_bw = inter_node_bandwidth_gbps if pp_crosses_nodes else intra_node_bandwidth_gbps
                pp_comm = (seq_len * dmodel * micro_batch * bytes_per_param * (pp - 1) * num_microbatches) / (pp_bw * 1e9 / 8) if pp > 1 else 0
                
                # Pipeline bubble overhead
                bubble_overhead = (pp - 1) / (num_microbatches + pp - 1) if pp > 1 else 0
                
                # Total estimated time
                score = compute_time * (1 + bubble_overhead) + tp_comm * 1000 + dp_comm * 1000 + pp_comm * 1000
                
                if score < best_score:
                    best_score = score
                    best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": micro_batch}
    
    if best_config is None:
        # Fallback: try aggressive memory saving with high TP and PP
        for tp in sorted(valid_tps, reverse=True):
            for pp in [8, 4, 2, 1]:
                if tp * pp > total_gpus:
                    continue
                dp = total_gpus // (tp * pp)
                if dp < 1:
                    continue
                min_mb = math.ceil(batch_size / dp)
                if min_mb < 1:
                    min_mb = 1
                best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": min_mb}
                break
            if best_config:
                break
    
    if best_config is None:
        # Ultimate fallback
        tp = min(valid_tps[-1], total_gpus)
        best_config = {"tp": tp, "dp": 1, "pp": 1, "micro_batch": batch_size}
    
    return best_config
