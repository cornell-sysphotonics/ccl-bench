
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
    Policy that balances memory constraints and communication overhead.
    
    Key considerations:
    1. Memory: larger models/sequences/batches need more TP and PP to fit
    2. Communication: TP is fast intra-node, PP adds bubble overhead, DP needs allreduce
    3. For large batch sizes, we need PP to reduce per-GPU memory
    """
    
    # Estimate model size per layer in bytes
    bytes_per_param = 4 if precision == "fp32" else 2
    
    # Rough parameter count per transformer layer:
    # 4 * dmodel^2 (QKV + output proj) + 8/3 * dmodel^2 (MLP with SwiGLU) ≈ ~12 * dmodel^2
    # For llama-style: Q,K,V projections + output + gate/up/down MLP
    params_per_layer = 12 * dmodel * dmodel  # approximate
    total_params = params_per_layer * num_stacks
    
    # Activation memory per sample per layer (rough estimate)
    # Main activations: seq_len * dmodel * bytes_per_param * some_factor
    # Factor accounts for attention scores, intermediate MLP, etc.
    activation_per_sample_per_layer = seq_len * dmodel * bytes_per_param * 10
    
    def estimate_memory_gb(tp, dp, pp, mb):
        """Estimate per-GPU memory usage in GB."""
        # Model parameters split across TP and PP
        model_mem = (total_params * bytes_per_param) / tp / pp
        
        # Number of layers per PP stage
        layers_per_stage = num_stacks / pp
        
        # Activation memory: depends on micro_batch size and layers per stage
        # With TP, activations are partially split
        act_mem = mb * activation_per_sample_per_layer * layers_per_stage / tp
        
        # Optimizer states (Adam: 2x model params for momentum + variance)
        optimizer_mem = model_mem * 2
        
        # Gradients
        grad_mem = model_mem
        
        total_mem = model_mem + optimizer_mem + grad_mem + act_mem
        return total_mem / (1024**3)
    
    def valid_tp(tp):
        return num_heads % tp == 0 and num_kv_heads % tp == 0 and tp <= gpus_per_node
    
    # Try different configurations and pick the best one that fits in memory
    best_config = None
    best_score = float('inf')
    
    # Possible TP values (must divide heads, prefer intra-node)
    tp_candidates = [t for t in [1, 2, 4, 8] if t <= gpus_per_node and valid_tp(t)]
    pp_candidates = [1, 2, 4, 8, 16]
    
    for tp in tp_candidates:
        for pp in pp_candidates:
            remaining = total_gpus // (tp * pp)
            if remaining < 1:
                continue
            dp = remaining
            
            if pp == 1:
                mb = batch_size // dp
                if mb < 1:
                    continue
                if batch_size % dp != 0:
                    continue
                
                mem = estimate_memory_gb(tp, dp, pp, mb)
                if mem > gpu_memory_gb * 0.9:  # 90% memory threshold
                    continue
                
                # Score: lower is better
                # TP communication cost (intra-node is fast)
                tp_comm = 0 if tp == 1 else (tp - 1) / tp * 2 * seq_len * dmodel * bytes_per_param * num_stacks / (intra_node_bandwidth_gbps * 1e9 / 8)
                
                # DP communication cost (allreduce of gradients)
                if dp > 1:
                    # Check if DP crosses nodes
                    dp_per_node = gpus_per_node // tp
                    if dp > dp_per_node:
                        dp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                    else:
                        dp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                    dp_comm = 2 * (dp - 1) / dp * total_params * bytes_per_param / tp / pp / dp_bw
                else:
                    dp_comm = 0
                
                # Compute time (rough: proportional to batch_size * seq_len * params / tp / dp)
                flops_per_gpu = 6 * total_params * batch_size * seq_len / (tp * dp)
                # A100 ~19.5 TFLOPS fp32, ~312 TFLOPS tf32/fp16
                gpu_flops = 19.5e12 if precision == "fp32" else 312e12
                compute_time = flops_per_gpu / gpu_flops
                
                score = compute_time + tp_comm + dp_comm
                
                if score < best_score:
                    best_score = score
                    best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
            
            else:
                # PP > 1: try different micro_batch sizes
                max_mb = batch_size // (dp * pp)
                if max_mb < 1:
                    continue
                if batch_size % (dp * pp) != 0:
                    continue
                
                for mb in range(1, max_mb + 1):
                    mem = estimate_memory_gb(tp, dp, pp, mb)
                    if mem > gpu_memory_gb * 0.9:
                        continue
                    
                    num_microbatches = batch_size // (dp * mb)
                    
                    # Pipeline bubble: (pp - 1) / num_microbatches fraction of time wasted
                    bubble_fraction = (pp - 1) / num_microbatches if num_microbatches > 0 else 1
                    
                    # Compute time per microbatch
                    flops_per_mb = 6 * (total_params / pp) * mb * seq_len / tp
                    gpu_flops = 19.5e12 if precision == "fp32" else 312e12
                    compute_per_mb = flops_per_mb / gpu_flops
                    
                    # Total compute with bubble
                    total_compute = compute_per_mb * num_microbatches * (1 + bubble_fraction)
                    
                    # TP communication
                    tp_comm = 0 if tp == 1 else (tp - 1) / tp * 2 * seq_len * dmodel * bytes_per_param * (num_stacks / pp) * num_microbatches / (intra_node_bandwidth_gbps * 1e9 / 8)
                    
                    # DP communication
                    if dp > 1:
                        dp_per_node = gpus_per_node // tp
                        if dp > dp_per_node:
                            dp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                        else:
                            dp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        dp_comm = 2 * (dp - 1) / dp * total_params * bytes_per_param / tp / pp / dp_bw
                    else:
                        dp_comm = 0
                    
                    # PP communication (send activations between stages)
                    pp_per_node = 1  # assume PP crosses nodes
                    pp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                    pp_comm = num_microbatches * mb * seq_len * dmodel * bytes_per_param / tp / pp_bw if pp > 1 else 0
                    
                    score = total_compute + tp_comm + dp_comm + pp_comm
                    
                    if score < best_score:
                        best_score = score
                        best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
    
    if best_config is None:
        # Fallback: maximum memory splitting
        tp = max(t for t in tp_candidates)
        pp = total_gpus // tp
        dp = 1
        mb = max(1, batch_size // (dp * pp))
        best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
    
    return best_config
