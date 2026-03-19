
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
    Improved policy focusing on:
    1. Fixing OOM/failures by being more conservative with memory estimates
    2. Ensuring batch_size is divisible by dp (when pp==1) or dp*mb divides batch_size (when pp>1)
    3. Optimizing wall time by balancing compute, communication, and pipeline bubbles
    """
    
    bytes_per_param = 4 if precision == "fp32" else 2
    
    # Parameter count per layer (llama-style):
    # Attention: Q(dmodel*dmodel) + K(dmodel*kv_dim) + V(dmodel*kv_dim) + O(dmodel*dmodel)
    # MLP (SwiGLU): gate(dmodel*4*dmodel/3*2) + down(4*dmodel/3*dmodel) ≈ 8/3*dmodel^2 * 2 + ... 
    # Simplified: ~12 * dmodel^2 per layer
    params_per_layer = 12 * dmodel * dmodel
    total_params = params_per_layer * num_stacks
    
    # More conservative activation memory estimate
    # Activations include: input embeddings, attention scores, intermediate MLP states
    # Per sample per layer: roughly seq_len * dmodel * bytes * factor
    # Factor ~12-16 to be conservative (attention scores are seq_len^2 * num_heads)
    activation_per_sample_per_layer = (
        seq_len * dmodel * bytes_per_param * 12 +  # linear layer activations
        seq_len * seq_len * num_heads * bytes_per_param  # attention scores
    )
    
    def estimate_memory_gb(tp, dp, pp, mb):
        """Estimate per-GPU memory usage in GB."""
        # Model parameters split across TP and PP
        model_mem = (total_params * bytes_per_param) / tp / pp
        
        layers_per_stage = num_stacks / pp
        
        # Activation memory: split by TP, scaled by micro_batch
        # Attention scores are split by TP (each head group)
        act_mem = mb * (
            seq_len * dmodel * bytes_per_param * 12 / tp +
            seq_len * seq_len * (num_heads / tp) * bytes_per_param
        ) * layers_per_stage
        
        # For PP > 1, we may need to store activations for multiple microbatches in flight
        if pp > 1:
            num_microbatches = batch_size // (dp * mb)
            # In 1F1B schedule, max in-flight is pp
            inflight = min(pp, num_microbatches)
            act_mem = act_mem * inflight / max(1, num_microbatches) * num_microbatches
            # Simplified: just use the base act_mem * min(pp, num_microbatches) / pp
            # Actually let's be more conservative: each stage stores activations for up to pp microbatches
            act_mem = mb * (
                seq_len * dmodel * bytes_per_param * 12 / tp +
                seq_len * seq_len * (num_heads / tp) * bytes_per_param
            ) * layers_per_stage * min(pp, num_microbatches)
        
        # Optimizer states (Adam: model + momentum + variance = 3x, but momentum+variance = 2x extra)
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
                
                # TP communication (2 allreduce per layer, each sending seq_len * dmodel * bytes)
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
                max_mb = batch_size // (dp * pp)
                if max_mb < 1:
                    continue
                    
                # Check batch_size divisibility
                for mb in range(1, max_mb + 1):
                    # batch_size must be divisible by dp * mb for clean microbatching
                    if batch_size % (dp * mb) != 0:
                        continue
                    
                    num_microbatches = batch_size // (dp * mb)
                    if num_microbatches < pp:
                        # Too few microbatches for pipeline to be efficient; but still valid
                        pass
                    
                    mem = estimate_memory_gb(tp, dp, pp, mb)
                    if mem > gpu_memory_gb * 0.85:
                        continue
                    
                    # Pipeline bubble
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
                        # Check if PP stages are on same node or cross nodes
                        # With tp GPUs per PP stage, and gpus_per_node GPUs per node
                        # If tp * pp <= gpus_per_node, PP is intra-node
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
        # Fallback: try to find anything that fits
        tp = max(tp_candidates)
        pp = total_gpus // tp
        dp = 1
        mb = 1
        best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
    
    return best_config
