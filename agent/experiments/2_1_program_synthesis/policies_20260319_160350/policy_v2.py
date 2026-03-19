
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
    Policy that balances memory constraints, communication overhead, and compute efficiency.
    
    Key insights from iteration 2:
    - tp=8 is too high, causes excessive communication overhead
    - For bs=128 with pp=2, micro_batch=1 causes timeout (too many pipeline steps)
    - Need to prefer smaller tp values and larger micro_batch for pp configs
    - Best score used avg_wall_time of ~270M, suggesting tp=4 or lower might be better
    """
    bytes_per_param = 4 if precision == "fp32" else 2
    
    # Estimate model parameters
    param_count = 12 * dmodel * dmodel * num_stacks
    
    # Memory per parameter (params + optimizer states + gradients)
    mem_per_param = 16  # bytes for fp32 Adam
    
    def estimate_memory_gb(tp, dp, pp, micro_batch):
        """Estimate per-GPU memory usage in GB."""
        # Model memory: split by tp and pp
        model_mem = (param_count * mem_per_param) / (tp * pp)
        
        # Activation memory per layer: roughly seq_len * dmodel * micro_batch * bytes * factor
        # Factor accounts for attention matrices, intermediate activations, etc.
        # Attention: micro_batch * num_heads/tp * seq_len * seq_len * bytes
        attn_act = micro_batch * (num_heads / tp) * seq_len * seq_len * bytes_per_param
        # FFN and other activations: micro_batch * seq_len * dmodel * bytes * factor
        ffn_act = micro_batch * seq_len * dmodel * bytes_per_param * 8
        
        layers_per_stage = num_stacks / pp
        activation_mem = (attn_act + ffn_act) * layers_per_stage
        
        # For pipeline parallelism, need to store activations for multiple micro-batches
        if pp > 1:
            activation_mem *= min(pp, 4)  # rough estimate of in-flight microbatches
        
        total_mem = model_mem + activation_mem
        return total_mem / (1024**3)
    
    def valid_tp(tp):
        return num_heads % tp == 0 and num_kv_heads % tp == 0
    
    best_config = None
    best_score = float('inf')
    
    possible_tp = [t for t in [1, 2, 4, 8] if t <= total_gpus and valid_tp(t)]
    possible_pp = [1, 2, 4, 8, 16, 32]
    
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
                
                # Compute time: proportional to micro_batch * seq_len^2 * dmodel * num_stacks / tp
                compute = micro_batch * seq_len * seq_len * dmodel * num_stacks / tp
                
                # TP communication cost (per layer): all-reduce of size seq_len * dmodel * micro_batch
                # More tp = more communication
                if tp <= gpus_per_node:
                    tp_bw = intra_node_bandwidth_gbps * 1e9 / 8  # bytes/sec
                else:
                    tp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                tp_comm = 2 * (tp - 1) / tp * (micro_batch * seq_len * dmodel * bytes_per_param) / tp_bw * num_stacks if tp > 1 else 0
                
                # DP communication: all-reduce gradients
                if dp > 1:
                    grad_size = param_count * bytes_per_param / (tp * pp)
                    # Check if dp spans nodes
                    gpus_in_dp_group = dp  # each dp rank is on different tp*pp groups
                    if total_gpus / gpus_per_node > 1 and dp > 1:
                        dp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                    else:
                        dp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                    dp_comm = 2 * (dp - 1) / dp * grad_size / dp_bw
                else:
                    dp_comm = 0
                
                # Score: weighted combination
                score = compute + (tp_comm + dp_comm) * 1e15
                
                if score < best_score:
                    best_score = score
                    best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": micro_batch}
                    
            else:  # pp > 1
                max_mb = global_batch_per_dp // pp
                if max_mb < 1:
                    continue
                
                # Try various micro batch sizes
                mb_candidates = set()
                for mb in range(1, max_mb + 1):
                    mb_candidates.add(mb)
                
                for mb in sorted(mb_candidates):
                    if mb < 1 or mb > max_mb:
                        continue
                    
                    mem = estimate_memory_gb(tp, dp, pp, mb)
                    if mem > gpu_memory_gb * 0.85:
                        continue
                    
                    num_microbatches = global_batch_per_dp // mb
                    if num_microbatches < pp:
                        # Too few microbatches for pipeline efficiency
                        continue
                    
                    # Pipeline bubble fraction
                    bubble_fraction = (pp - 1) / (num_microbatches + pp - 1)
                    
                    # Compute per microbatch per stage
                    compute_per_mb = mb * seq_len * seq_len * dmodel * (num_stacks / pp) / tp
                    # Total compute with bubble
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
                        if total_gpus / gpus_per_node > 1 and dp > 1:
                            dp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                        else:
                            dp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        dp_comm = 2 * (dp - 1) / dp * grad_size / dp_bw
                    else:
                        dp_comm = 0
                    
                    # PP communication: send activations between stages
                    if pp > 1:
                        act_size = mb * seq_len * dmodel * bytes_per_param
                        # Check if pp stages cross nodes
                        if tp * pp > gpus_per_node:
                            pp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                        else:
                            pp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        pp_comm = act_size / pp_bw * (num_microbatches + pp - 1) * 2  # fwd + bwd
                    else:
                        pp_comm = 0
                    
                    score = total_compute + (tp_comm + dp_comm + pp_comm) * 1e15
                    
                    if score < best_score:
                        best_score = score
                        best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
    
    if best_config is None:
        # Fallback: use maximum tp and pp to fit in memory
        tp = min(max(possible_tp), gpus_per_node)
        pp = max(2, total_gpus // (tp * 2))
        dp = max(1, total_gpus // (tp * pp))
        mb = max(1, batch_size // (dp * pp))
        best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
    
    return best_config
