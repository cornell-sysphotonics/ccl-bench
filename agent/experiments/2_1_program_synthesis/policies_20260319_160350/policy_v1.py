
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
    Improved policy that handles OOM by using pipeline parallelism when needed,
    and tries to minimize wall time through better parallelism choices.
    """
    # Bytes per parameter based on precision
    bytes_per_param = 4 if precision == "fp32" else 2  # fp16/bf16 = 2 bytes
    
    # Estimate model parameters (rough estimate for transformer models)
    # For a transformer: ~12 * dmodel^2 * num_stacks parameters
    param_count = 12 * dmodel * dmodel * num_stacks
    
    # Memory per parameter (params + optimizer states + gradients)
    # For Adam: params(4B) + gradients(4B) + optimizer(8B) = 16B per param in fp32
    # For mixed precision: params(2B) + master(4B) + gradients(2B) + optimizer(8B) = 16B
    mem_per_param = 16  # bytes, conservative estimate
    
    # Activation memory estimate per layer per sample:
    # ~= seq_len * dmodel * batch_size * bytes_per_param * some_factor
    # This is a rough estimate; activation checkpointing might help
    
    def estimate_memory_gb(tp, dp, pp, micro_batch):
        """Estimate per-GPU memory usage in GB."""
        # Model memory: split by tp and pp
        model_mem = (param_count * mem_per_param) / (tp * pp)
        
        # Activation memory: proportional to micro_batch * seq_len * dmodel
        # Split by tp, and only 1/pp of layers per stage
        activation_mem = (micro_batch * seq_len * dmodel * bytes_per_param * 20 * num_stacks) / (tp * pp)
        
        total_mem = model_mem + activation_mem
        return total_mem / (1024**3)  # Convert to GB
    
    def valid_tp(tp):
        return num_heads % tp == 0 and num_kv_heads % tp == 0
    
    # Strategy: try different configurations and pick the one that should be fastest
    # Priority: maximize dp (data parallelism) while fitting in memory
    # Use tp for memory reduction within a node, pp as last resort
    
    best_config = None
    best_score = float('inf')
    
    # Possible tp values (must divide heads and kv_heads, and <= gpus_per_node ideally)
    possible_tp = [t for t in [1, 2, 4, 8] if t <= total_gpus and valid_tp(t)]
    possible_pp = [1, 2, 4, 8, 16]
    
    for tp in possible_tp:
        for pp in possible_pp:
            if tp * pp > total_gpus:
                continue
            dp = total_gpus // (tp * pp)
            if dp < 1:
                continue
            
            if pp == 1:
                if batch_size % dp != 0:
                    continue
                micro_batch = batch_size // dp
                
                # Check memory
                mem = estimate_memory_gb(tp, dp, pp, micro_batch)
                if mem > gpu_memory_gb * 0.9:  # 90% threshold
                    continue
                
                # Estimate wall time score (lower is better)
                # Communication cost for tp (all-reduce within node if tp <= gpus_per_node)
                if tp <= gpus_per_node:
                    tp_comm = tp * dmodel * seq_len * bytes_per_param / (intra_node_bandwidth_gbps * 1e9 / 8)
                else:
                    tp_comm = tp * dmodel * seq_len * bytes_per_param / (inter_node_bandwidth_gbps * 1e9 / 8)
                
                # DP communication (all-reduce of gradients)
                if dp <= 1:
                    dp_comm = 0
                else:
                    # Gradient size split by tp and pp
                    grad_size = param_count * bytes_per_param / (tp * pp)
                    # Ring all-reduce: 2*(dp-1)/dp * grad_size / bandwidth
                    nodes_in_dp = dp * tp // gpus_per_node  # rough
                    if dp * tp <= gpus_per_node:
                        bw = intra_node_bandwidth_gbps * 1e9 / 8
                    else:
                        bw = inter_node_bandwidth_gbps * 1e9 / 8
                    dp_comm = 2 * (dp - 1) / dp * grad_size / bw
                
                # Compute time proportional to micro_batch * seq_len^2 * dmodel * num_stacks / tp
                compute = micro_batch * seq_len * seq_len * dmodel * num_stacks / tp
                
                score = compute + (tp_comm + dp_comm) * 1e6  # rough weighting
                
            else:  # pp > 1
                if batch_size % dp != 0:
                    continue
                global_batch_per_dp = batch_size // dp
                
                # Try different micro batch sizes
                best_mb = None
                best_pp_score = float('inf')
                
                for mb in range(1, global_batch_per_dp // pp + 1):
                    if global_batch_per_dp % mb != 0:
                        continue
                    
                    mem = estimate_memory_gb(tp, dp, pp, mb)
                    if mem > gpu_memory_gb * 0.9:
                        continue
                    
                    num_microbatches = global_batch_per_dp // mb
                    # Pipeline bubble: (pp - 1) / num_microbatches fraction of time wasted
                    bubble_fraction = (pp - 1) / (num_microbatches + pp - 1)
                    
                    compute = mb * seq_len * seq_len * dmodel * num_stacks / (tp * pp)
                    compute_total = compute * num_microbatches / (1 - bubble_fraction) if bubble_fraction < 1 else float('inf')
                    
                    if tp <= gpus_per_node:
                        tp_comm = tp * dmodel * seq_len * bytes_per_param / (intra_node_bandwidth_gbps * 1e9 / 8)
                    else:
                        tp_comm = tp * dmodel * seq_len * bytes_per_param / (inter_node_bandwidth_gbps * 1e9 / 8)
                    
                    if dp <= 1:
                        dp_comm = 0
                    else:
                        grad_size = param_count * bytes_per_param / (tp * pp)
                        if dp * tp <= gpus_per_node:
                            bw = intra_node_bandwidth_gbps * 1e9 / 8
                        else:
                            bw = inter_node_bandwidth_gbps * 1e9 / 8
                        dp_comm = 2 * (dp - 1) / dp * grad_size / bw
                    
                    pp_score = compute_total + (tp_comm + dp_comm) * 1e6
                    
                    if pp_score < best_pp_score:
                        best_pp_score = pp_score
                        best_mb = mb
                
                if best_mb is None:
                    continue
                
                score = best_pp_score
                micro_batch = best_mb
            
            if pp == 1:
                micro_batch_final = batch_size // dp
            else:
                micro_batch_final = best_mb
            
            if score < best_score:
                best_score = score
                best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": micro_batch_final}
    
    if best_config is None:
        # Fallback: aggressive memory saving
        tp = min(max(possible_tp), gpus_per_node)
        pp = 2
        dp = total_gpus // (tp * pp)
        if dp < 1:
            dp = 1
            pp = total_gpus // tp
        micro_batch = max(1, batch_size // (dp * pp))
        best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": micro_batch}
    
    return best_config
