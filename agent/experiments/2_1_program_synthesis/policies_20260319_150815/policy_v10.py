
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
    import math
    
    bytes_per_param = 4 if precision == "fp32" else 2
    
    # Valid TP values must divide both num_heads and num_kv_heads
    valid_tps = []
    for t in [1, 2, 4, 8, 16, 32, 64]:
        if t <= total_gpus and num_heads % t == 0 and num_kv_heads % t == 0:
            valid_tps.append(t)
    
    best_config = None
    best_score = float('inf')
    
    for tp in valid_tps:
        # Prefer TP within a node
        if tp > gpus_per_node:
            continue  # Skip TP crossing nodes - too expensive
            
        for pp in [1, 2, 4, 8, 16, 32]:
            if tp * pp > total_gpus:
                continue
            
            dp = total_gpus // (tp * pp)
            if dp < 1:
                continue
            if tp * dp * pp > total_gpus:
                continue
            
            # micro_batch must be >= ceil(batch_size / dp)
            min_mb = math.ceil(batch_size / dp)
            if min_mb < 1:
                min_mb = 1
            
            # Try several micro_batch sizes
            mb_candidates = set()
            mb_candidates.add(min_mb)
            for val in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                if val >= min_mb and val <= batch_size:
                    mb_candidates.add(val)
            for mult in [1, 2, 3, 4, 8]:
                val = min_mb * mult
                if val <= batch_size:
                    mb_candidates.add(val)
            # Also try min_mb + small increments
            for inc in range(0, 5):
                val = min_mb + inc
                if val <= batch_size:
                    mb_candidates.add(val)
            
            for micro_batch in sorted(mb_candidates):
                if micro_batch < min_mb:
                    continue
                
                # Estimate memory per GPU
                layers_per_pp = math.ceil(num_stacks / pp)
                
                # Model parameters per layer: ~12 * dmodel^2 for a standard transformer
                params_per_layer = 12 * dmodel * dmodel
                params_per_gpu = params_per_layer * layers_per_pp / tp
                
                # Model memory
                model_mem = params_per_gpu * bytes_per_param
                
                # Optimizer memory (Adam): params + momentum + variance in fp32
                if precision == "fp32":
                    opt_mem = params_per_gpu * 4 * 2  # momentum + variance
                else:
                    opt_mem = params_per_gpu * 4 * 3  # master weights + momentum + variance
                
                # Gradient memory
                grad_mem = params_per_gpu * bytes_per_param
                
                # Activation memory per micro-batch per layer
                attn_score_mem = micro_batch * (num_heads / tp) * seq_len * seq_len * bytes_per_param
                linear_act_mem = 10 * seq_len * dmodel * micro_batch * bytes_per_param / tp
                ffn_act_mem = 2 * seq_len * 4 * dmodel * micro_batch * bytes_per_param / tp
                
                act_per_layer = attn_score_mem + linear_act_mem + ffn_act_mem
                act_mem = act_per_layer * layers_per_pp
                
                # For pipeline parallelism, need to store activations for multiple micro-batches
                if pp > 1:
                    num_microbatches = math.ceil(batch_size / (dp * micro_batch))
                    stored_mb = min(pp, num_microbatches)
                    act_mem = act_mem * stored_mb
                
                total_mem = model_mem + opt_mem + grad_mem + act_mem
                total_mem_gb = total_mem / (1024**3)
                
                # Memory check with margin
                if total_mem_gb > gpu_memory_gb * 0.85:
                    continue
                
                # --- Estimate wall time ---
                num_microbatches = math.ceil(batch_size / (dp * micro_batch))
                
                # Compute per GPU per microbatch
                compute_per_mb = 6 * params_per_layer * layers_per_pp * micro_batch * seq_len / tp
                total_compute = compute_per_mb * num_microbatches
                
                # A100 peak FLOPS
                peak_flops = 156e12 if precision == "fp32" else 312e12
                compute_efficiency = 0.40
                compute_time = total_compute / (peak_flops * compute_efficiency)
                
                # Pipeline bubble overhead
                if pp > 1:
                    bubble_fraction = (pp - 1) / (num_microbatches + pp - 1)
                    if bubble_fraction >= 0.95:
                        # Too much bubble, skip
                        continue
                    compute_time_with_bubble = compute_time / (1 - bubble_fraction)
                else:
                    bubble_fraction = 0
                    compute_time_with_bubble = compute_time
                
                # TP communication (all-reduce within node)
                if tp > 1:
                    tp_bw_bytes = intra_node_bandwidth_gbps * 1e9 / 8
                    tp_msg_size = seq_len * dmodel * micro_batch * bytes_per_param / tp
                    # 2 all-reduces per layer (one for attn, one for FFN), each with 2*(tp-1)/tp ring cost
                    tp_comm_per_layer = 2 * 2 * (tp - 1) / tp * tp_msg_size / tp_bw_bytes
                    tp_comm = tp_comm_per_layer * layers_per_pp * num_microbatches
                    tp_latency = 5e-6 * 4 * layers_per_pp * num_microbatches
                    tp_comm += tp_latency
                else:
                    tp_comm = 0
                
                # DP communication (gradient all-reduce)
                if dp > 1:
                    # Check if DP crosses nodes
                    # With tp GPUs per DP rank, if we have more DP ranks than can fit on nodes
                    dp_crosses_nodes = (dp * tp > gpus_per_node) or (total_gpus > gpus_per_node)
                    if dp_crosses_nodes:
                        dp_bw_bytes = inter_node_bandwidth_gbps * 1e9 / 8
                    else:
                        dp_bw_bytes = intra_node_bandwidth_gbps * 1e9 / 8
                    dp_msg_size = params_per_gpu * bytes_per_param
                    dp_comm = 2 * (dp - 1) / dp * dp_msg_size / dp_bw_bytes
                    dp_latency = 50e-6 if dp_crosses_nodes else 5e-6
                    dp_comm += dp_latency * 10
                else:
                    dp_comm = 0
                
                # PP communication
                if pp > 1:
                    # Check if PP crosses nodes
                    pp_crosses_nodes = (pp > 1) and (total_gpus > gpus_per_node)
                    if pp_crosses_nodes:
                        pp_bw_bytes = inter_node_bandwidth_gbps * 1e9 / 8
                    else:
                        pp_bw_bytes = intra_node_bandwidth_gbps * 1e9 / 8
                    pp_msg_size = seq_len * dmodel * micro_batch * bytes_per_param
                    # Point-to-point sends between pipeline stages
                    pp_comm = pp_msg_size * 2 * num_microbatches / pp_bw_bytes
                    pp_latency = 50e-6 if pp_crosses_nodes else 5e-6
                    pp_comm += pp_latency * 2 * num_microbatches
                    if pp_crosses_nodes:
                        pp_comm *= 2
                else:
                    pp_comm = 0
                
                # Total estimated wall time
                # DP comm can overlap with backward pass
                dp_overlap = 0.7 if dp > 1 else 0
                score = compute_time_with_bubble + tp_comm + dp_comm * (1 - dp_overlap) + pp_comm
                
                if score < best_score:
                    best_score = score
                    best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": micro_batch}
    
    # If nothing found (e.g., everything OOM), try with TP crossing nodes
    if best_config is None:
        for tp in valid_tps:
            for pp in [1, 2, 4, 8, 16, 32]:
                if tp * pp > total_gpus:
                    continue
                dp = total_gpus // (tp * pp)
                if dp < 1:
                    continue
                min_mb = math.ceil(batch_size / dp)
                if min_mb < 1:
                    min_mb = 1
                
                layers_per_pp = math.ceil(num_stacks / pp)
                params_per_layer = 12 * dmodel * dmodel
                params_per_gpu = params_per_layer * layers_per_pp / tp
                model_mem = params_per_gpu * bytes_per_param
                opt_mem = params_per_gpu * 4 * 2
                grad_mem = params_per_gpu * bytes_per_param
                
                attn_score_mem = min_mb * (num_heads / tp) * seq_len * seq_len * bytes_per_param
                linear_act_mem = 10 * seq_len * dmodel * min_mb * bytes_per_param / tp
                ffn_act_mem = 2 * seq_len * 4 * dmodel * min_mb * bytes_per_param / tp
                act_per_layer = attn_score_mem + linear_act_mem + ffn_act_mem
                act_mem = act_per_layer * layers_per_pp
                if pp > 1:
                    num_microbatches = math.ceil(batch_size / (dp * min_mb))
                    stored_mb = min(pp, num_microbatches)
                    act_mem = act_mem * stored_mb
                
                total_mem_gb = (model_mem + opt_mem + grad_mem + act_mem) / (1024**3)
                if total_mem_gb <= gpu_memory_gb * 0.90:
                    best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": min_mb}
                    break
            if best_config:
                break
    
    if best_config is None:
        tp = valid_tps[-1] if valid_tps else 1
        best_config = {"tp": tp, "dp": 1, "pp": 1, "micro_batch": batch_size}
    
    return best_config
