
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
    Iteration 5 analysis:
    - Almost everything FAILED with tp=4,dp=2,pp=4 due to timeout.
    - The pp>1 configs keep timing out, even with pp=4 and seemingly reasonable steps.
    - Best previous result was iteration 2 with 80% success at tp=4,dp=8,pp=1.
    
    Strategy: GO BACK TO BASICS.
    - Use pp=1 as strongly as possible. tp=4,dp=8,pp=1 worked for 4/5 workloads.
    - Only the seq2048 case might OOM with pp=1. For that, try tp=4,dp=4,pp=2 or similar.
    - If pp=1 doesn't fit in memory, try pp=2 with LARGE micro_batch.
    - NEVER use pp>=4 - it keeps timing out.
    
    The key insight: with pp=1, micro_batch = batch_size/dp (fixed), and there's no
    pipeline bubble. This is fast and reliable.
    
    For the seq2048 OOM case: the issue is activation memory doubles with seq_len.
    Options: increase tp (tp=4 already), or use pp=2 with careful config.
    Actually, let me just try tp=4,dp=8,pp=1 for everything and see what happens.
    If seq2048 OOMs, then I need pp=2 for that case only.
    """
    bytes_per_param = 4 if precision == "fp32" else 2
    
    def valid_tp(tp):
        return num_heads % tp == 0 and num_kv_heads % tp == 0
    
    # Strategy: try pp=1 configs first, then pp=2 as fallback
    # For pp=1: try different tp values, maximize dp for throughput
    
    candidates = []
    
    # pp=1 candidates (strongly preferred)
    for tp in [1, 2, 4, 8]:
        if tp > total_gpus or tp > gpus_per_node:
            continue
        if not valid_tp(tp):
            continue
        
        dp = total_gpus // tp
        pp = 1
        
        if dp < 1:
            continue
        if batch_size % dp != 0:
            # Try smaller dp
            for d in range(dp, 0, -1):
                if batch_size % d == 0 and tp * d <= total_gpus:
                    dp = d
                    break
            else:
                continue
        
        micro_batch = batch_size // dp
        
        # Rough memory estimate
        param_count = 12 * dmodel * dmodel * num_stacks
        model_mem_bytes = param_count * 16 / tp  # params + optimizer + grads (fp32)
        if precision != "fp32":
            model_mem_bytes = param_count * 12 / tp
        
        # Activation memory
        attn_act = micro_batch * (num_heads // tp) * seq_len * seq_len * bytes_per_param
        ffn_act = micro_batch * seq_len * dmodel * bytes_per_param * 8
        act_mem = (attn_act + ffn_act) * num_stacks
        
        total_mem_gb = (model_mem_bytes + act_mem) / (1024**3)
        
        if total_mem_gb > gpu_memory_gb * 0.85:
            continue
        
        # Score this config
        # Compute time (proportional)
        compute = micro_batch * seq_len * num_stacks / tp
        
        # TP comm cost
        tp_comm = 0
        if tp > 1:
            tp_comm = (tp - 1) / tp * dmodel * bytes_per_param * 2 * num_stacks * micro_batch * seq_len
            if tp <= gpus_per_node:
                tp_comm /= (intra_node_bandwidth_gbps * 1e9 / 8)
            else:
                tp_comm /= (inter_node_bandwidth_gbps * 1e9 / 8)
        
        # DP comm cost (all-reduce gradients)
        dp_comm = 0
        if dp > 1:
            grad_size = param_count * bytes_per_param / tp
            # DP spans nodes if dp > gpus_per_node/tp
            dp_per_node = gpus_per_node // tp
            if dp <= dp_per_node:
                bw = intra_node_bandwidth_gbps * 1e9 / 8
            else:
                bw = inter_node_bandwidth_gbps * 1e9 / 8
            dp_comm = 2 * (dp - 1) / dp * grad_size / bw
        
        # Higher dp = more parallelism = lower compute time per GPU
        # But also more communication
        score = compute + (tp_comm + dp_comm) * 1e12
        
        candidates.append((score, {"tp": tp, "dp": dp, "pp": pp, "micro_batch": micro_batch}))
    
    # pp=2 candidates (fallback for memory-constrained cases)
    for tp in [1, 2, 4]:
        if tp > gpus_per_node:
            continue
        if not valid_tp(tp):
            continue
        
        pp = 2
        if tp * pp > total_gpus:
            continue
        
        dp = total_gpus // (tp * pp)
        if dp < 1:
            continue
        
        # Adjust dp if batch_size not divisible
        for d in range(dp, 0, -1):
            if batch_size % d == 0 and tp * d * pp <= total_gpus:
                dp = d
                break
        else:
            continue
        
        global_batch_per_dp = batch_size // dp
        if global_batch_per_dp < pp:
            continue
        
        max_mb = global_batch_per_dp // pp
        
        # Use largest micro_batch possible to minimize pipeline steps
        micro_batch = max_mb
        if micro_batch < 1:
            continue
        
        num_microbatches = global_batch_per_dp // micro_batch
        total_steps = num_microbatches + pp - 1
        
        # Memory estimate with pp
        param_count = 12 * dmodel * dmodel * num_stacks
        model_mem_bytes = param_count * 16 / (tp * pp)
        if precision != "fp32":
            model_mem_bytes = param_count * 12 / (tp * pp)
        
        layers_per_stage = num_stacks // pp
        attn_act = micro_batch * (num_heads // tp) * seq_len * seq_len * bytes_per_param
        ffn_act = micro_batch * seq_len * dmodel * bytes_per_param * 8
        act_mem = (attn_act + ffn_act) * layers_per_stage * min(pp, num_microbatches)
        
        total_mem_gb = (model_mem_bytes + act_mem) / (1024**3)
        
        if total_mem_gb > gpu_memory_gb * 0.85:
            continue
        
        # Score - add penalty for pp>1
        compute = micro_batch * seq_len * layers_per_stage / tp * total_steps
        bubble_penalty = (pp - 1) / total_steps
        compute *= (1 + bubble_penalty)
        
        score = compute * 1.5  # penalty for using pipeline parallelism
        
        candidates.append((score, {"tp": tp, "dp": dp, "pp": pp, "micro_batch": micro_batch}))
    
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    # Ultimate fallback
    tp = 4
    while tp > 1 and not valid_tp(tp):
        tp //= 2
    dp = total_gpus // tp
    while dp > 1 and batch_size % dp != 0:
        dp -= 1
    return {"tp": tp, "dp": dp, "pp": 1, "micro_batch": batch_size // dp}
