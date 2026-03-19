
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
    Iteration 4 analysis:
    - llama-8b-bs64 (tp=4,dp=1,pp=8,mb=2) FAILED: timeout. bs64 with dp=1,pp=8 means
      global_batch_per_dp=64, num_microbatches=64/2=32, total_steps=32+7=39. Still too many.
    - llama-8b-bs128 (tp=2,dp=2,pp=8,mb=2) FAILED: timeout. global_batch_per_dp=64,
      num_microbatches=32, total_steps=39. Same issue.
    
    Best previous iteration (80%, 269M): used tp=4,dp=8,pp=1 for most workloads.
    
    Strategy for this iteration:
    - STRONGLY prefer pp=1 configs (no pipeline bubble, no timeout risk)
    - Only use pp>1 when absolutely needed for memory, with large micro_batch
    - For pp>1, limit total pipeline steps to ~20
    - Key: tp=4,dp=8,pp=1 worked for 4/5 workloads before. Need to figure out
      which one failed and fix it.
    
    Let me think about what configs work:
    - tp=4,dp=8,pp=1: For bs32, mb=32/8=4. For bs64, mb=64/8=8. For bs128, mb=128/8=16.
      For bs16, mb=16/8=2. For seq2048, mb=32/8=4.
    - The one that failed before with tp=4,dp=8,pp=1 was likely an OOM case.
    
    Let me try tp=4,dp=8,pp=1 as default and handle OOM cases carefully.
    Actually, let me try to be smarter about memory estimation.
    """
    bytes_per_param = 4 if precision == "fp32" else 2
    
    # Estimate model parameters
    param_count = 12 * dmodel * dmodel * num_stacks
    
    # Memory per parameter (params + optimizer states + gradients)
    mem_per_param = 16 if precision == "fp32" else 12
    
    def estimate_memory_gb(tp, dp, pp, micro_batch):
        """Estimate per-GPU memory usage in GB."""
        # Model memory: split by tp and pp
        model_mem = (param_count * mem_per_param) / (tp * pp)
        
        # Activation memory per layer
        attn_act = micro_batch * (num_heads / tp) * seq_len * seq_len * bytes_per_param
        ffn_act = micro_batch * seq_len * dmodel * bytes_per_param * 8
        
        layers_per_stage = num_stacks / pp if pp > 1 else num_stacks
        activation_mem = (attn_act + ffn_act) * layers_per_stage
        
        # For pipeline parallelism, need to store activations for multiple micro-batches
        if pp > 1:
            activation_mem *= min(pp, 4)
        
        total_mem = model_mem + activation_mem
        return total_mem / (1024**3)
    
    def valid_tp(tp):
        return num_heads % tp == 0 and num_kv_heads % tp == 0
    
    best_config = None
    best_score = float('inf')
    
    possible_tp = [t for t in [1, 2, 4] if t <= total_gpus and t <= gpus_per_node and valid_tp(t)]
    if not possible_tp:
        possible_tp = [t for t in [1, 2, 4, 8] if t <= total_gpus and valid_tp(t)]
    
    # Try pp=1 first (strongly preferred), then pp=2, pp=4
    # Avoid pp=8 entirely - too many pipeline stages cause timeouts
    possible_pp = [1, 2, 4]
    
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
                
                # Score: prioritize pp=1 configs
                # Compute time estimate (relative)
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
                    grad_size = param_count * bytes_per_param / tp
                    # Check if dp group spans nodes
                    # With tp GPUs per tp group, gpus_per_node/tp dp ranks can be intra-node
                    dp_per_node = gpus_per_node // tp
                    if dp <= dp_per_node:
                        dp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                    else:
                        dp_bw = inter_node_bandwidth_gbps * 1e9 / 8
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
                
                # Try micro batch sizes from large to small
                mb_candidates = set()
                for mb in range(max(1, max_mb), 0, -1):
                    mb_candidates.add(mb)
                    if len(mb_candidates) >= 8:
                        break
                for mb in [1, 2, 4, 8]:
                    if mb <= max_mb:
                        mb_candidates.add(mb)
                
                for mb in sorted(mb_candidates, reverse=True):
                    if mb < 1 or mb > max_mb:
                        continue
                    
                    num_microbatches = global_batch_per_dp // mb
                    
                    # Total pipeline steps
                    total_steps = num_microbatches + pp - 1
                    
                    # STRICT limit on pipeline steps to avoid timeout
                    if total_steps > 25:
                        continue
                    
                    # Want enough microbatches relative to pp for efficiency
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
                    # Account for bubble overhead
                    total_compute *= (1 + bubble_fraction * 0.5)
                    
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
                        dp_per_node = gpus_per_node // tp
                        if dp <= dp_per_node:
                            dp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                        else:
                            dp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                        dp_comm = 2 * (dp - 1) / dp * grad_size / dp_bw
                    else:
                        dp_comm = 0
                    
                    # PP communication
                    act_size = mb * seq_len * dmodel * bytes_per_param
                    # Check if pp stages cross nodes
                    stages_per_node = gpus_per_node // tp
                    if pp <= stages_per_node:
                        pp_bw = intra_node_bandwidth_gbps * 1e9 / 8
                    else:
                        pp_bw = inter_node_bandwidth_gbps * 1e9 / 8
                    pp_comm = act_size / pp_bw * (num_microbatches + pp - 1) * 2
                    
                    score = total_compute + (tp_comm + dp_comm + pp_comm) * 1e15
                    
                    if score < best_score:
                        best_score = score
                        best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": mb}
    
    if best_config is None:
        # Fallback: simple config
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
