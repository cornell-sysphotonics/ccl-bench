
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
    Key insights from iteration 5 results:
    - llama-8b-bs32 with tp=4,dp=4,pp=2,mb=1 FAILED (no metrics)
    - llama-8b-bs128 with tp=4,dp=4,pp=2,mb=1 FAILED (timeout - too many pipeline steps)
    - Best previous score was 80% with avg 240M wall time
    
    Strategy:
    1. Prefer pp=1 configurations to avoid pipeline bubble issues
    2. Use TP within node for memory reduction
    3. Use DP for batch parallelism
    4. Only use PP when absolutely needed for memory, and with larger micro_batch
    5. Better memory estimation to know when pp=1 is feasible
    """
    
    bytes_per_param = 4 if precision == "fp32" else 2
    
    # Parameter count estimation for llama-style models
    # Per layer: Q,K,V,O projections + MLP (gate, up, down with SwiGLU)
    # Q: dmodel * dmodel, K: dmodel * (dmodel * num_kv_heads/num_heads), 
    # V: same as K, O: dmodel * dmodel
    # MLP with SwiGLU: gate: dmodel * 4*dmodel, up: dmodel * 4*dmodel, down: 4*dmodel * dmodel
    # Plus layer norms, embeddings etc.
    kv_dim = dmodel * num_kv_heads // num_heads
    params_per_layer = (dmodel * dmodel +  # Q
                        dmodel * kv_dim +   # K
                        dmodel * kv_dim +   # V
                        dmodel * dmodel +   # O
                        dmodel * 4 * dmodel +  # gate
                        dmodel * 4 * dmodel +  # up
                        4 * dmodel * dmodel)   # down
    # ~= 12 * dmodel^2 for standard case where kv_dim = dmodel
    
    total_params = params_per_layer * num_stacks
    # Add embedding layer
    vocab_size = 32000  # typical for llama
    embedding_params = vocab_size * dmodel
    
    def estimate_memory_gb(tp, dp, pp, mb):
        """Estimate per-GPU memory usage in GB."""
        layers_per_stage = num_stacks // pp if pp > 1 else num_stacks
        
        # Model parameters split across TP and PP
        model_params = params_per_layer * layers_per_stage / tp
        model_mem_bytes = model_params * bytes_per_param
        
        # Optimizer states (Adam: params + momentum + variance)
        if precision == "fp32":
            optimizer_mem = model_mem_bytes * 3  # momentum + variance + copy
        else:
            optimizer_mem = model_mem_bytes * 4  # fp32 master weights + momentum + variance
        
        # Gradients
        grad_mem = model_mem_bytes
        
        # Activation memory per microbatch
        # Hidden states, attention intermediates, MLP intermediates
        # Rough estimate: per layer, we need to store:
        # - Input hidden states: mb * seq_len * dmodel
        # - Attention scores: mb * num_heads/tp * seq_len * seq_len
        # - MLP intermediates: mb * seq_len * 4 * dmodel / tp
        # Factor of ~10-20x for all intermediates including backward
        
        act_per_layer_per_mb = (
            mb * seq_len * dmodel * bytes_per_param * 10 / tp +  # hidden states & intermediates
            mb * (num_heads // tp) * seq_len * seq_len * bytes_per_param * 2  # attention scores (fwd+bwd)
        )
        
        if pp > 1:
            num_microbatches = batch_size // (dp * mb)
            # 1F1B: need to store min(pp, num_microbatches) activations
            inflight = min(pp, num_microbatches)
            act_mem = act_per_layer_per_mb * layers_per_stage * inflight
        else:
            act_mem = act_per_layer_per_mb * layers_per_stage
        
        total_mem = model_mem_bytes + optimizer_mem + grad_mem + act_mem
        return total_mem / (1024**3)
    
    def valid_tp(tp):
        return num_heads % tp == 0 and num_kv_heads % tp == 0 and tp <= gpus_per_node
    
    gpu_flops = 19.5e12 if precision == "fp32" else 312e12
    
    tp_candidates = [t for t in [1, 2, 4, 8] if t <= gpus_per_node and valid_tp(t)]
    
    best_config = None
    best_score = float('inf')
    
    mem_limit = gpu_memory_gb * 0.85  # 85% memory utilization limit
    
    # Strategy: enumerate configs, preferring pp=1
    # PP candidates ordered to prefer pp=1
    pp_candidates = [1, 2, 4, 8, 16, 32]
    
    for tp in tp_candidates:
        for pp in pp_candidates:
            if tp * pp > total_gpus:
                continue
            
            # Check pp divides num_stacks
            if num_stacks % pp != 0:
                continue
            
            remaining = total_gpus // (tp * pp)
            
            # Try different dp values (not just max)
            dp_candidates = set()
            for dp in range(1, remaining + 1):
                if batch_size % dp == 0:
                    dp_candidates.add(dp)
            # Always include the maximum dp
            if batch_size % remaining == 0:
                dp_candidates.add(remaining)
            
            for dp in sorted(dp_candidates, reverse=True):  # prefer larger dp
                if dp < 1 or dp > remaining:
                    continue
                if batch_size % dp != 0:
                    continue
                
                local_batch = batch_size // dp
                
                if pp == 1:
                    mb = local_batch
                    if mb < 1:
                        continue
                    
                    mem = estimate_memory_gb(tp, dp, pp, mb)
                    if mem > mem_limit:
                        continue
                    
                    # Compute time
                    flops_per_gpu = 6 * total_params * mb * seq_len / tp
                    compute_time = flops_per_gpu / gpu_flops
                    
                    # TP communication
                    if tp > 1:
                        tp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        msg_size = mb * seq_len * dmodel * bytes_per_param
                        # 2 allreduce per layer (fwd + bwd), each allreduce = 2*(tp-1)/tp * msg
                        tp_comm = 4 * num_stacks * 2 * (tp - 1) / tp * msg_size / tp_bw
                    else:
                        tp_comm = 0
                    
                    # DP communication
                    if dp > 1:
                        # Check if DP is within or across nodes
                        gpus_used_per_node = tp  # tp GPUs per node for this config
                        dp_per_node = gpus_per_node // tp
                        if dp <= dp_per_node:
                            dp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        else:
                            dp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                        grad_size = total_params * bytes_per_param / tp
                        dp_comm = 2 * (dp - 1) / dp * grad_size / dp_bw
                    else:
                        dp_comm = 0
                    
                    score = compute_time + tp_comm + dp_comm
                    
                    if score < best_score:
                        best_score = score
                        best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
                
                else:
                    # PP > 1
                    if local_batch < 1:
                        continue
                    
                    # Try various micro_batch sizes
                    mb_candidates = set()
                    for mb in range(1, local_batch + 1):
                        if local_batch % mb == 0:
                            num_microbatches = local_batch // mb
                            total_steps = num_microbatches + pp - 1
                            # Strict limit to avoid timeout
                            if total_steps <= 50:
                                mb_candidates.add(mb)
                    
                    if not mb_candidates:
                        # If nothing works with step limit, try largest possible mb
                        for mb in range(local_batch, 0, -1):
                            if local_batch % mb == 0:
                                mb_candidates.add(mb)
                                break
                    
                    for mb in mb_candidates:
                        if mb < 1:
                            continue
                        num_microbatches = local_batch // mb
                        if num_microbatches < 1:
                            continue
                        
                        # Check constraint: micro_batch <= batch_size / (dp * pp)
                        max_mb_allowed = batch_size // (dp * pp)
                        if max_mb_allowed < 1:
                            continue
                        if mb > max_mb_allowed:
                            continue
                        
                        total_steps = num_microbatches + pp - 1
                        if total_steps > 60:
                            continue
                        
                        mem = estimate_memory_gb(tp, dp, pp, mb)
                        if mem > mem_limit:
                            continue
                        
                        layers_per_stage = num_stacks // pp
                        
                        # Pipeline bubble fraction
                        bubble_fraction = (pp - 1) / num_microbatches
                        
                        # Compute time per microbatch per stage
                        flops_per_mb = 6 * params_per_layer * layers_per_stage * mb * seq_len / tp
                        compute_per_mb = flops_per_mb / gpu_flops
                        
                        # Total compute with bubble
                        total_compute = compute_per_mb * (num_microbatches + pp - 1)
                        
                        # TP communication
                        if tp > 1:
                            tp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                            msg_size = mb * seq_len * dmodel * bytes_per_param
                            tp_comm = 4 * layers_per_stage * num_microbatches * 2 * (tp - 1) / tp * msg_size / tp_bw
                        else:
                            tp_comm = 0
                        
                        # DP communication
                        if dp > 1:
                            dp_per_node = gpus_per_node // tp
                            if dp <= dp_per_node:
                                dp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                            else:
                                dp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                            grad_size = total_params * bytes_per_param / tp / pp
                            dp_comm = 2 * (dp - 1) / dp * grad_size / dp_bw
                        else:
                            dp_comm = 0
                        
                        # PP communication
                        if tp * pp <= gpus_per_node:
                            pp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        else:
                            pp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                        activation_size = mb * seq_len * dmodel * bytes_per_param / tp
                        pp_comm = num_microbatches * 2 * activation_size / pp_bw * (pp - 1)
                        
                        score = total_compute + tp_comm + dp_comm + pp_comm
                        
                        if score < best_score:
                            best_score = score
                            best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
    
    if best_config is None:
        # Aggressive fallback
        for tp in sorted(tp_candidates, reverse=True):
            for pp in sorted(pp_candidates, reverse=True):
                if tp * pp > total_gpus:
                    continue
                if num_stacks % pp != 0:
                    continue
                dp = 1
                if batch_size % dp != 0:
                    continue
                local_batch = batch_size // dp
                
                if pp == 1:
                    mb = local_batch
                else:
                    mb = max(1, local_batch // pp)
                    # Find valid mb
                    while mb >= 1 and local_batch % mb != 0:
                        mb -= 1
                    if mb < 1:
                        mb = 1
                
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
