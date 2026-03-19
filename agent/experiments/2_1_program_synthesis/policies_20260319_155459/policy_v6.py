
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
    Iteration 6 analysis:
    - llama-8b-bs32 with tp=4,dp=4,pp=2,mb=1 FAILED (no metrics) - likely OOM or invalid config
    - llama-8b-bs128 with tp=4,dp=4,pp=2,mb=1 FAILED (timeout) - too many pipeline steps
    - llama-8b-bs64 with tp=4,dp=4,pp=2,mb=1 = 897M (very slow)
    - llama-8b-seq2048 with tp=4,dp=4,pp=2,mb=1 = 738M (slow)
    - llama-8b-bs16-seq512 with tp=4,dp=1,pp=8,mb=1 = 92M (ok but could be better)
    
    Best historical results (iteration 3, 80% success, avg 240M):
    - llama-8b-bs32: tp=4,dp=2,pp=1,mb=16 -> 240M
    - llama-8b-bs64: tp=4,dp=2,pp=1,mb=32 -> 410M  
    - llama-8b-seq2048: tp=4,dp=2,pp=1,mb=16 -> 376M
    - llama-8b-bs16-seq512: tp=4,dp=2,pp=1,mb=8 -> 136M
    - llama-8b-bs128: FAILED (OOM with tp=4,dp=2,pp=1)
    
    Key insight: pp=1 configs with tp=4,dp=2 worked well for most cases.
    For bs128, we need pp>1 since per-GPU batch is too large.
    
    Strategy:
    1. Default to tp=4, dp=2, pp=1 for moderate sizes (proven to work)
    2. For large batch (bs128): use pp=2 with larger micro_batch to avoid timeout
    3. For bs32 with tp=4,dp=4,pp=2: this config failed - need different approach
       bs32: with tp=4,dp=2,pp=1 -> mb=16 (worked before at 240M)
    
    Let me be more careful about memory estimation and prefer pp=1 strongly.
    """
    
    bytes_per_param = 4 if precision == "fp32" else 2
    
    # Parameter estimation
    kv_dim = dmodel * num_kv_heads // num_heads
    params_per_layer = (dmodel * dmodel +  # Q
                        dmodel * kv_dim +   # K
                        dmodel * kv_dim +   # V
                        dmodel * dmodel +   # O
                        dmodel * 4 * dmodel +  # gate (SwiGLU)
                        dmodel * 4 * dmodel +  # up
                        4 * dmodel * dmodel)   # down
    
    total_params = params_per_layer * num_stacks
    vocab_size = 32000
    embedding_params = vocab_size * dmodel
    
    def estimate_memory_gb(tp, dp, pp, mb):
        """Estimate per-GPU memory usage in GB."""
        layers_per_stage = num_stacks // pp if pp > 1 else num_stacks
        
        # Model parameters split across TP and PP
        model_params = params_per_layer * layers_per_stage / tp
        model_mem_bytes = model_params * bytes_per_param
        
        # Optimizer states (Adam: params + momentum + variance = 3x)
        optimizer_mem = model_mem_bytes * 3
        
        # Gradients
        grad_mem = model_mem_bytes
        
        # Activation memory
        # Per layer per sample: hidden states + attention scores + MLP intermediates
        act_per_layer = (
            mb * seq_len * dmodel * bytes_per_param * 12 / tp +  # hidden states & intermediates  
            mb * (num_heads // tp) * seq_len * seq_len * bytes_per_param * 2  # attention scores
        )
        
        if pp > 1:
            num_microbatches = batch_size // (dp * mb)
            inflight = min(pp, num_microbatches)
            act_mem = act_per_layer * layers_per_stage * inflight
        else:
            act_mem = act_per_layer * layers_per_stage
        
        total_mem = model_mem_bytes + optimizer_mem + grad_mem + act_mem
        return total_mem / (1024**3)
    
    def valid_tp(tp):
        return num_heads % tp == 0 and num_kv_heads % tp == 0 and tp <= gpus_per_node
    
    gpu_flops = 19.5e12 if precision == "fp32" else 312e12
    
    tp_candidates = [t for t in [1, 2, 4, 8] if t <= gpus_per_node and valid_tp(t)]
    
    best_config = None
    best_score = float('inf')
    
    mem_limit = gpu_memory_gb * 0.82  # conservative memory limit
    
    # Enumerate configurations, strongly preferring pp=1
    pp_candidates = [1, 2, 4, 8]
    
    for pp in pp_candidates:
        for tp in sorted(tp_candidates, reverse=True):  # prefer larger tp first for memory
            if tp * pp > total_gpus:
                continue
            if num_stacks % pp != 0:
                continue
            
            remaining = total_gpus // (tp * pp)
            
            # Try different dp values
            dp_candidates = []
            for dp in range(1, remaining + 1):
                if batch_size % dp == 0:
                    dp_candidates.append(dp)
            
            for dp in sorted(dp_candidates, reverse=True):  # prefer larger dp
                if dp > remaining:
                    continue
                
                local_batch = batch_size // dp
                
                if pp == 1:
                    mb = local_batch
                    if mb < 1:
                        continue
                    
                    mem = estimate_memory_gb(tp, dp, pp, mb)
                    if mem > mem_limit:
                        continue
                    
                    # Compute score
                    flops_per_gpu = 6 * total_params * mb * seq_len / tp
                    compute_time = flops_per_gpu / gpu_flops
                    
                    # TP communication
                    if tp > 1:
                        tp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        msg_size = mb * seq_len * dmodel * bytes_per_param
                        tp_comm = 4 * num_stacks * 2 * (tp - 1) / tp * msg_size / tp_bw
                    else:
                        tp_comm = 0
                    
                    # DP communication
                    if dp > 1:
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
                    max_mb_allowed = batch_size // (dp * pp)
                    if max_mb_allowed < 1:
                        continue
                    
                    # Try various micro_batch sizes, prefer larger to reduce pipeline steps
                    mb_candidates = []
                    for mb in range(max_mb_allowed, 0, -1):
                        if local_batch % mb == 0:
                            num_microbatches = local_batch // mb
                            total_steps = num_microbatches + pp - 1
                            if total_steps <= 40:  # strict limit to avoid timeout
                                mb_candidates.append(mb)
                    
                    if not mb_candidates:
                        # Try the largest possible mb even if many steps
                        for mb in range(max_mb_allowed, 0, -1):
                            if local_batch % mb == 0:
                                mb_candidates.append(mb)
                                break
                    
                    for mb in mb_candidates:
                        num_microbatches = local_batch // mb
                        if num_microbatches < 1:
                            continue
                        
                        total_steps = num_microbatches + pp - 1
                        if total_steps > 50:
                            continue
                        
                        mem = estimate_memory_gb(tp, dp, pp, mb)
                        if mem > mem_limit:
                            continue
                        
                        layers_per_stage = num_stacks // pp
                        
                        # Compute time per microbatch per stage
                        flops_per_mb = 6 * params_per_layer * layers_per_stage * mb * seq_len / tp
                        compute_per_mb = flops_per_mb / gpu_flops
                        
                        # Total compute with bubble
                        total_compute = compute_per_mb * total_steps
                        
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
                            grad_size = total_params * bytes_per_param / (tp * pp)
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
                        
                        # Add a penalty for pp>1 to prefer pp=1 when close
                        pp_penalty = 1.1 if pp > 1 else 1.0
                        
                        score = (total_compute + tp_comm + dp_comm + pp_comm) * pp_penalty
                        
                        if score < best_score:
                            best_score = score
                            best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
    
    if best_config is None:
        # Fallback: try aggressive memory configs
        for tp in sorted(tp_candidates, reverse=True):
            for pp in [1, 2, 4, 8, 16]:
                if tp * pp > total_gpus:
                    continue
                if num_stacks % pp != 0:
                    continue
                dp = 1
                local_batch = batch_size
                if pp == 1:
                    mb = local_batch
                else:
                    max_mb_allowed = batch_size // (dp * pp)
                    if max_mb_allowed < 1:
                        continue
                    mb = max_mb_allowed
                
                mem = estimate_memory_gb(tp, dp, pp, mb)
                if mem <= gpu_memory_gb * 0.95:
                    best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
                    break
            if best_config:
                break
        
        if best_config is None:
            tp = max(tp_candidates)
            best_config = {"tp": tp, "dp": 1, "pp": total_gpus // tp, "micro_batch": 1}
    
    return best_config
