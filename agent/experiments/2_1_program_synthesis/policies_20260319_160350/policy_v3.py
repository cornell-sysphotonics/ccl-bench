
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
    Key lessons from iterations 1-3:
    - micro_batch=1 with large pp causes timeouts (too many pipeline steps)
    - tp=8 causes excessive communication overhead
    - Best previous result was success_rate=80% with avg_wall_time=269M
      That used tp=4,dp=8,pp=1 for most workloads
    - Need to avoid pp>1 with micro_batch=1 (causes timeouts)
    - Priority: maximize success rate first, then minimize wall time
    
    Strategy: Prefer pp=1 configs. Only use pp>1 when memory requires it,
    and ensure micro_batch >= 2 when using pp.
    """
    bytes_per_param = 4 if precision == "fp32" else 2
    
    # Estimate model parameters (rough)
    param_count = 12 * dmodel * dmodel * num_stacks
    
    # Memory per parameter (params + optimizer states + gradients)
    # fp32 Adam: param(4) + grad(4) + m(4) + v(4) = 16 bytes
    mem_per_param = 16 if precision == "fp32" else 12
    
    def estimate_memory_gb(tp, dp, pp, micro_batch):
        """Estimate per-GPU memory usage in GB."""
        # Model memory: split by tp and pp
        model_mem = (param_count * mem_per_param) / (tp * pp)
        
        # Activation memory per layer
        # Attention: micro_batch * num_heads/tp * seq_len * seq_len * bytes
        attn_act = micro_batch * (num_heads / tp) * seq_len * seq_len * bytes_per_param
        # FFN and other activations: micro_batch * seq_len * dmodel * bytes * factor
        ffn_act = micro_batch * seq_len * dmodel * bytes_per_param * 8
        
        layers_per_stage = num_stacks / pp if pp > 1 else num_stacks
        activation_mem = (attn_act + ffn_act) * layers_per_stage
        
        # For pipeline parallelism, need to store activations for multiple micro-batches in flight
        if pp > 1:
            activation_mem *= min(pp, 4)
        
        total_mem = model_mem + activation_mem
        return total_mem / (1024**3)
    
    def valid_tp(tp):
        return num_heads % tp == 0 and num_kv_heads % tp == 0
    
    def spans_nodes(group_size, tp_val=1):
        """Check if a parallelism group spans multiple nodes."""
        # With tp_val GPUs per tp group, dp/pp groups may span nodes
        # Simple heuristic: if total_gpus > gpus_per_node, some groups span nodes
        return total_gpus > gpus_per_node and group_size > 1
    
    best_config = None
    best_score = float('inf')
    
    possible_tp = [t for t in [1, 2, 4, 8] if t <= total_gpus and t <= gpus_per_node and valid_tp(t)]
    if not possible_tp:
        possible_tp = [t for t in [1, 2, 4, 8] if t <= total_gpus and valid_tp(t)]
    
    possible_pp = [1, 2, 4, 8]
    
    for tp in possible_tp:
        for pp in possible_pp:
            if tp * pp > total_gpus:
                continue
            dp = total_gpus // (tp * pp)
            if dp < 1:
                continue
            if batch_size % dp != 0:
                continue
            
            global_batch_per_dp = batch_size // dp
            
            if pp == 1:
                micro_batch = global_batch_per_dp
                
                mem = estimate_memory_gb(tp, dp, pp, micro_batch)
                if mem > gpu_memory_gb * 0.85:
                    continue
                
                # Compute time estimate
                compute = micro_batch * seq_len * seq_len * dmodel * num_stacks / tp
                
                # TP communication cost
                if tp > 1:
                    if tp <= gpus_per_node:
                        tp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                    else:
                        tp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                    tp_comm = 2 * (tp - 1) / tp * (micro_batch * seq_len * dmodel * bytes_per_param) / tp_bw * num_stacks
                else:
                    tp_comm = 0
                
                # DP communication: all-reduce gradients
                if dp > 1:
                    grad_size = param_count * bytes_per_param / (tp * pp)
                    if spans_nodes(dp):
                        dp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                    else:
                        dp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                    dp_comm = 2 * (dp - 1) / dp * grad_size / dp_bw
                else:
                    dp_comm = 0
                
                score = compute + (tp_comm + dp_comm) * 1e15
                
                if score < best_score:
                    best_score = score
                    best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": micro_batch}
                    
            else:  # pp > 1
                if global_batch_per_dp < pp:
                    continue
                
                max_mb = global_batch_per_dp // pp
                if max_mb < 1:
                    continue
                
                # Try various micro batch sizes - prefer larger ones to avoid too many pipeline steps
                mb_candidates = []
                for mb in range(max(1, max_mb), 0, -1):
                    mb_candidates.append(mb)
                    if len(mb_candidates) > 10:
                        break
                # Also add small values
                for mb in [1, 2, 4]:
                    if mb <= max_mb and mb not in mb_candidates:
                        mb_candidates.append(mb)
                
                for mb in mb_candidates:
                    if mb < 1 or mb > max_mb:
                        continue
                    
                    num_microbatches = global_batch_per_dp // mb
                    
                    # CRITICAL: Avoid configs with too many pipeline steps (causes timeout)
                    # Total steps ~ num_microbatches + pp - 1
                    total_steps = num_microbatches + pp - 1
                    if total_steps > 50:
                        continue
                    
                    # Also avoid micro_batch=1 with large pp (known timeout issue)
                    if mb == 1 and pp >= 4:
                        continue
                    
                    if num_microbatches < pp:
                        continue
                    
                    mem = estimate_memory_gb(tp, dp, pp, mb)
                    if mem > gpu_memory_gb * 0.85:
                        continue
                    
                    # Pipeline bubble fraction
                    bubble_fraction = (pp - 1) / (num_microbatches + pp - 1)
                    
                    # Compute per microbatch per stage
                    compute_per_mb = mb * seq_len * seq_len * dmodel * (num_stacks / pp) / tp
                    total_compute = compute_per_mb * (num_microbatches + pp - 1)
                    
                    # TP communication
                    if tp > 1:
                        if tp <= gpus_per_node:
                            tp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        else:
                            tp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                        tp_comm = 2 * (tp - 1) / tp * (mb * seq_len * dmodel * bytes_per_param) / tp_bw * (num_stacks / pp) * (num_microbatches + pp - 1)
                    else:
                        tp_comm = 0
                    
                    # DP communication
                    if dp > 1:
                        grad_size = param_count * bytes_per_param / (tp * pp)
                        if spans_nodes(dp):
                            dp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                        else:
                            dp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        dp_comm = 2 * (dp - 1) / dp * grad_size / dp_bw
                    else:
                        dp_comm = 0
                    
                    # PP communication
                    act_size = mb * seq_len * dmodel * bytes_per_param
                    if tp * pp > gpus_per_node or pp > gpus_per_node // tp:
                        pp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                    else:
                        pp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                    pp_comm = act_size / pp_bw * (num_microbatches + pp - 1) * 2
                    
                    score = total_compute + (tp_comm + dp_comm + pp_comm) * 1e15
                    
                    if score < best_score:
                        best_score = score
                        best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
    
    if best_config is None:
        # Fallback: simple config that should work
        tp = min(4, gpus_per_node)
        while tp > 1 and not valid_tp(tp):
            tp //= 2
        dp = total_gpus // tp
        while dp > 1 and batch_size % dp != 0:
            dp -= 1
        pp = 1
        micro_batch = batch_size // dp
        best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": micro_batch}
    
    return best_config
