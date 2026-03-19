
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
    1. Fixing OOM failures by better memory estimation
    2. Fixing timeout failures by limiting pipeline steps
    3. Preferring pp=1 when possible (simpler, no bubble overhead)
    4. Using TP within node, DP across nodes when possible
    """
    
    bytes_per_param = 4 if precision == "fp32" else 2
    
    # Parameter count per layer (llama-style transformer)
    # Q, K, V, O projections + MLP (gate, up, down with SwiGLU)
    # More accurate: Q=dmodel^2, K=dmodel*(dmodel*num_kv_heads/num_heads), V=same, O=dmodel^2
    # MLP with SwiGLU: gate=dmodel*4*dmodel, up=dmodel*4*dmodel, down=4*dmodel*dmodel
    # Approximate: ~12 * dmodel^2 per layer for standard llama
    params_per_layer = 12 * dmodel * dmodel
    total_params = params_per_layer * num_stacks
    
    def estimate_memory_gb(tp, dp, pp, mb):
        """Estimate per-GPU memory usage in GB - conservative."""
        # Model parameters split across TP and PP
        model_mem = (total_params * bytes_per_param) / tp / pp
        
        layers_per_stage = num_stacks / pp
        
        # Activation memory per microbatch per stage
        # Key components: hidden states, attention scores, MLP intermediates
        act_per_mb = mb * (
            # Hidden states through the layers (conservative: ~16x factor for all intermediates)
            seq_len * dmodel * bytes_per_param * 16 / tp +
            # Attention score matrix
            seq_len * seq_len * (num_heads / tp) * bytes_per_param
        ) * layers_per_stage
        
        if pp > 1:
            num_microbatches = batch_size // (dp * mb)
            # In 1F1B schedule, need to store activations for min(pp, num_microbatches) microbatches
            inflight = min(pp, num_microbatches)
            act_mem = act_per_mb * inflight
        else:
            act_mem = act_per_mb
        
        # Optimizer states (Adam: params + momentum + variance = 3x, plus master weights for fp16)
        optimizer_mem = model_mem * 3 if precision == "fp32" else model_mem * 4
        
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
    
    mem_fraction = 0.80  # Conservative memory limit
    
    for tp in tp_candidates:
        for pp in pp_candidates:
            if tp * pp > total_gpus:
                continue
            remaining = total_gpus // (tp * pp)
            if remaining < 1:
                continue
            dp = remaining
            
            if batch_size % dp != 0:
                continue
            local_batch = batch_size // dp
            
            if pp == 1:
                mb = local_batch
                if mb < 1:
                    continue
                
                mem = estimate_memory_gb(tp, dp, pp, mb)
                if mem > gpu_memory_gb * mem_fraction:
                    continue
                
                # Compute time
                flops_per_gpu = 6 * total_params * mb * seq_len / tp
                compute_time = flops_per_gpu / gpu_flops
                
                # TP communication (2 allreduce per layer: forward + backward)
                if tp > 1:
                    tp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                    # Each allreduce: 2*(tp-1)/tp * message_size
                    msg_size = mb * seq_len * dmodel * bytes_per_param
                    tp_comm = 4 * num_stacks * 2 * (tp - 1) / tp * msg_size / tp_bw
                else:
                    tp_comm = 0
                
                # DP communication (allreduce gradients)
                if dp > 1:
                    dp_per_node = gpus_per_node // tp
                    nodes_in_dp = (dp + dp_per_node - 1) // dp_per_node
                    if nodes_in_dp > 1:
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
                if local_batch < 1:
                    continue
                
                # Generate candidate micro_batch sizes
                max_mb = local_batch // pp
                if max_mb < 1:
                    max_mb = 1
                
                mb_candidates = set()
                for mb in range(1, min(max_mb + 1, local_batch + 1)):
                    if local_batch % mb == 0:
                        num_microbatches = local_batch // mb
                        if num_microbatches >= pp:  # Need at least pp microbatches for good efficiency
                            mb_candidates.add(mb)
                
                # Also try mb values that give exactly pp microbatches (good balance)
                if local_batch % pp == 0:
                    mb_candidates.add(local_batch // pp)
                
                # And 2*pp microbatches
                if local_batch % (2 * pp) == 0:
                    mb_candidates.add(local_batch // (2 * pp))
                
                if not mb_candidates:
                    # Try mb=1 as fallback
                    mb_candidates = {1}
                
                for mb in mb_candidates:
                    if mb < 1 or local_batch % mb != 0:
                        continue
                    num_microbatches = local_batch // mb
                    if num_microbatches < 1:
                        continue
                    
                    # Hard limit on total pipeline steps to avoid timeout
                    total_steps = num_microbatches + pp - 1
                    if total_steps > 100:
                        continue
                    
                    mem = estimate_memory_gb(tp, dp, pp, mb)
                    if mem > gpu_memory_gb * mem_fraction:
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
                        msg_size = mb * seq_len * dmodel * bytes_per_param
                        tp_comm = 4 * layers_per_stage * num_microbatches * 2 * (tp - 1) / tp * msg_size / tp_bw
                    else:
                        tp_comm = 0
                    
                    # DP communication
                    if dp > 1:
                        dp_per_node = gpus_per_node // tp
                        nodes_in_dp = (dp + dp_per_node - 1) // dp_per_node
                        if nodes_in_dp > 1:
                            dp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                        else:
                            dp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        grad_size = total_params * bytes_per_param / tp / pp
                        dp_comm = 2 * (dp - 1) / dp * grad_size / dp_bw
                    else:
                        dp_comm = 0
                    
                    # PP communication (point-to-point between stages)
                    if pp > 1:
                        # Check if PP stages are within a node or across nodes
                        if tp * pp <= gpus_per_node:
                            pp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        else:
                            pp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                        # Each microbatch sends activation between stages (forward + backward)
                        activation_size = mb * seq_len * dmodel * bytes_per_param / tp
                        pp_comm = num_microbatches * 2 * activation_size / pp_bw
                    else:
                        pp_comm = 0
                    
                    score = total_compute + tp_comm + dp_comm + pp_comm
                    
                    if score < best_score:
                        best_score = score
                        best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
    
    if best_config is None:
        # Fallback: try increasingly aggressive configs to find anything that fits
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
                    # Find a valid mb
                    mb = 1
                    for m in range(1, local_batch + 1):
                        if local_batch % m == 0:
                            num_mb = local_batch // m
                            if num_mb + pp - 1 <= 100:
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
