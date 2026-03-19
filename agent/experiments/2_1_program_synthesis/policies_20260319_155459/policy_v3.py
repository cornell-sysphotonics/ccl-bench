
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
    """
    Policy focusing on:
    1. Fixing failures: avoid too-small micro_batch with large pp (causes timeout)
    2. Fixing "no metrics" errors (likely OOM or invalid config)
    3. Preferring larger micro_batch to reduce pipeline steps
    4. Being more careful about memory estimates
    """
    
    bytes_per_param = 4 if precision == "fp32" else 2
    
    # Parameter count per layer (llama-style transformer)
    # Q, K, V, O projections + MLP (gate, up, down with SwiGLU)
    # Approximate: ~12 * dmodel^2 per layer
    params_per_layer = 12 * dmodel * dmodel
    total_params = params_per_layer * num_stacks
    
    def estimate_memory_gb(tp, dp, pp, mb):
        """Estimate per-GPU memory usage in GB."""
        # Model parameters split across TP and PP
        model_mem = (total_params * bytes_per_param) / tp / pp
        
        layers_per_stage = num_stacks / pp
        
        # Activation memory per microbatch per stage
        # Include attention scores (seq_len^2 * num_heads/tp) and layer activations
        act_per_mb = mb * (
            seq_len * dmodel * bytes_per_param * 14 / tp +
            seq_len * seq_len * (num_heads / tp) * bytes_per_param
        ) * layers_per_stage
        
        if pp > 1:
            num_microbatches = batch_size // (dp * mb)
            # In 1F1B schedule, need to store activations for min(pp, num_microbatches) microbatches
            inflight = min(pp, num_microbatches)
            act_mem = act_per_mb * inflight
        else:
            act_mem = act_per_mb
        
        # Optimizer states (Adam: momentum + variance = 2x model params)
        optimizer_mem = model_mem * 2
        
        # Gradients
        grad_mem = model_mem
        
        total_mem = model_mem + optimizer_mem + grad_mem + act_mem
        return total_mem / (1024**3)
    
    def valid_tp(tp):
        return num_heads % tp == 0 and num_kv_heads % tp == 0 and tp <= gpus_per_node
    
    best_config = None
    best_score = float('inf')
    
    tp_candidates = [t for t in [1, 2, 4, 8] if t <= gpus_per_node and valid_tp(t)]
    pp_candidates = [1, 2, 4, 8, 16, 32]
    
    gpu_flops = 19.5e12 if precision == "fp32" else 312e12
    
    for tp in tp_candidates:
        for pp in pp_candidates:
            if tp * pp > total_gpus:
                continue
            remaining = total_gpus // (tp * pp)
            if remaining < 1:
                continue
            dp = remaining
            
            if pp == 1:
                # micro_batch is fixed = batch_size / dp
                if batch_size % dp != 0:
                    continue
                mb = batch_size // dp
                if mb < 1:
                    continue
                
                mem = estimate_memory_gb(tp, dp, pp, mb)
                if mem > gpu_memory_gb * 0.85:
                    continue
                
                # Compute time
                flops_per_gpu = 6 * total_params * mb * seq_len / tp
                compute_time = flops_per_gpu / gpu_flops
                
                # TP communication
                if tp > 1:
                    tp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                    tp_comm = 2 * (tp - 1) / tp * seq_len * dmodel * bytes_per_param * num_stacks * mb / tp_bw
                else:
                    tp_comm = 0
                
                # DP communication (allreduce gradients)
                if dp > 1:
                    dp_per_node = gpus_per_node // tp
                    if dp > dp_per_node:
                        dp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                    else:
                        dp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                    grad_size = total_params * bytes_per_param / tp / pp
                    dp_comm = 2 * (dp - 1) / dp * grad_size / dp_bw
                else:
                    dp_comm = 0
                
                score = compute_time + tp_comm + dp_comm
                
                if score < best_score:
                    best_score = score
                    best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
            
            else:
                # PP > 1
                if batch_size % dp != 0:
                    continue
                local_batch = batch_size // dp
                
                # Try different micro_batch sizes, prefer larger ones to reduce pipeline steps
                max_mb = local_batch // pp
                if max_mb < 1:
                    # Can still try mb=1 if local_batch >= 1
                    max_mb = local_batch
                    if max_mb < 1:
                        continue
                
                # Generate candidate micro_batch sizes
                mb_candidates = []
                for mb in range(1, max_mb + 1):
                    if local_batch % mb == 0:
                        num_microbatches = local_batch // mb
                        if num_microbatches >= 1:
                            mb_candidates.append(mb)
                
                if not mb_candidates:
                    continue
                    
                for mb in mb_candidates:
                    num_microbatches = local_batch // mb
                    
                    # Skip configs that would cause too many pipeline steps (timeout risk)
                    # Total pipeline steps ~ num_microbatches + pp - 1
                    # If this is very large, simulation may time out
                    total_steps = num_microbatches + pp - 1
                    if total_steps > 200:
                        continue
                    
                    # Need at least pp microbatches for reasonable pipeline efficiency
                    # (but don't skip, just penalize)
                    
                    mem = estimate_memory_gb(tp, dp, pp, mb)
                    if mem > gpu_memory_gb * 0.85:
                        continue
                    
                    # Pipeline bubble fraction
                    bubble_fraction = (pp - 1) / num_microbatches if num_microbatches > 0 else 1
                    
                    # Compute time per microbatch
                    layers_per_stage = num_stacks / pp
                    flops_per_mb = 6 * params_per_layer * layers_per_stage * mb * seq_len / tp
                    compute_per_mb = flops_per_mb / gpu_flops
                    
                    # Total compute with bubble
                    total_compute = compute_per_mb * num_microbatches * (1 + bubble_fraction)
                    
                    # TP communication
                    if tp > 1:
                        tp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        tp_comm = 2 * (tp - 1) / tp * seq_len * dmodel * bytes_per_param * layers_per_stage * num_microbatches * mb / tp_bw
                    else:
                        tp_comm = 0
                    
                    # DP communication
                    if dp > 1:
                        dp_per_node = gpus_per_node // tp
                        if dp > dp_per_node:
                            dp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                        else:
                            dp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        grad_size = total_params * bytes_per_param / tp / pp
                        dp_comm = 2 * (dp - 1) / dp * grad_size / dp_bw
                    else:
                        dp_comm = 0
                    
                    # PP communication
                    if pp > 1:
                        if tp * pp <= gpus_per_node:
                            pp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        else:
                            pp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                        pp_comm = num_microbatches * 2 * mb * seq_len * dmodel * bytes_per_param / tp / pp_bw
                    else:
                        pp_comm = 0
                    
                    score = total_compute + tp_comm + dp_comm + pp_comm
                    
                    if score < best_score:
                        best_score = score
                        best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
    
    if best_config is None:
        # Fallback: try increasingly aggressive configs
        for tp in sorted(tp_candidates, reverse=True):
            for pp in sorted(pp_candidates, reverse=True):
                if tp * pp > total_gpus:
                    continue
                dp = total_gpus // (tp * pp)
                if dp < 1:
                    continue
                if batch_size % dp != 0:
                    continue
                local_batch = batch_size // dp
                if pp == 1:
                    mb = local_batch
                else:
                    # Find largest valid mb
                    mb = 1
                    for m in range(local_batch, 0, -1):
                        if local_batch % m == 0:
                            mb = m
                            break
                
                mem = estimate_memory_gb(tp, dp, pp, mb)
                if mem <= gpu_memory_gb * 0.95:
                    best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
                    break
            if best_config:
                break
        
        if best_config is None:
            tp = max(tp_candidates)
            pp = total_gpus // tp
            dp = 1
            mb = 1
            best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
    
    return best_config
