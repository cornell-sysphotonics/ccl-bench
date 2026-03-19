
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
    import math
    
    bytes_per_param = 4 if precision == "fp32" else 2
    
    # Valid TP values must divide both num_heads and num_kv_heads
    valid_tps = []
    for t in [1, 2, 4, 8, 16, 32]:
        if t <= total_gpus and num_heads % t == 0 and num_kv_heads % t == 0:
            valid_tps.append(t)
    
    best_config = None
    best_score = float('inf')
    
    for tp in valid_tps:
        for pp in [1, 2, 4, 8, 16, 32]:
            if tp * pp > total_gpus:
                continue
            
            dp = total_gpus // (tp * pp)
            if dp < 1:
                continue
            if tp * dp * pp > total_gpus:
                continue
            
            # micro_batch must be >= ceil(batch_size / dp)
            min_mb = math.ceil(batch_size / dp)
            if min_mb < 1:
                min_mb = 1
            
            # Try several micro_batch sizes
            mb_candidates = set()
            mb_candidates.add(min_mb)
            for mult in [2, 4, 8]:
                val = min_mb * mult
                if val <= batch_size:
                    mb_candidates.add(val)
            # Also try batch_size itself
            mb_candidates.add(batch_size // dp if batch_size // dp >= min_mb else min_mb)
            
            for micro_batch in sorted(mb_candidates):
                if micro_batch < min_mb:
                    continue
                
                # Estimate memory per GPU
                layers_per_pp = math.ceil(num_stacks / pp)
                
                # Model parameters per layer: ~12 * dmodel^2
                params_per_layer = 12 * dmodel * dmodel
                params_per_gpu = params_per_layer * layers_per_pp / tp
                
                # Model memory
                model_mem = params_per_gpu * bytes_per_param
                
                # Optimizer memory: params + momentum + variance (all fp32) + gradients
                # Adam: 3 copies in fp32 (param copy, momentum, variance) + gradients
                opt_mem = params_per_gpu * 4 * 3  # optimizer states always fp32
                grad_mem = params_per_gpu * bytes_per_param
                
                # Activation memory per micro-batch per layer
                # Key components: attention scores, intermediate FFN, layer norms, etc.
                # Rough: activation_factor * seq_len * dmodel * micro_batch * bytes / tp
                activation_factor = 34
                act_mem = (seq_len * dmodel * micro_batch * bytes_per_param * activation_factor * layers_per_pp) / tp
                
                # For pipeline parallelism, we may need to store activations for multiple micro-batches
                # In 1F1B schedule, max stored = pp micro-batches
                if pp > 1:
                    num_microbatches = math.ceil(batch_size / (dp * micro_batch))
                    stored_mb = min(pp, num_microbatches)
                    act_mem = act_mem * stored_mb
                
                total_mem = model_mem + opt_mem + grad_mem + act_mem
                total_mem_gb = total_mem / (1024**3)
                
                # Memory check with safety margin
                if total_mem_gb > gpu_memory_gb * 0.80:
                    continue
                
                # Estimate wall time (heuristic scoring)
                num_microbatches = math.ceil(batch_size / (dp * micro_batch))
                
                # Compute time proportional to work per GPU
                compute_time = micro_batch * seq_len * params_per_gpu * num_microbatches
                
                # TP communication: all-reduce after each layer, 2 per layer (attention + FFN)
                tp_crosses_nodes = tp > gpus_per_node
                tp_bw = inter_node_bandwidth_gbps if tp_crosses_nodes else intra_node_bandwidth_gbps
                # Message size for all-reduce: seq_len * dmodel * micro_batch * bytes
                tp_msg_size = seq_len * dmodel * micro_batch * bytes_per_param
                tp_comm = (tp_msg_size * 2 * (tp - 1) / tp * layers_per_pp * num_microbatches) / (tp_bw * 1e9 / 8) if tp > 1 else 0
                
                # DP communication: all-reduce gradients once
                # Check if DP crosses nodes
                dp_within_node = (tp * pp <= gpus_per_node) and (dp <= gpus_per_node // (tp * pp) if tp * pp <= gpus_per_node else False)
                # More accurate: figure out if dp replicas span nodes
                gpus_used_per_node = min(gpus_per_node, tp * pp)
                dp_per_node = gpus_per_node // gpus_used_per_node if gpus_used_per_node <= gpus_per_node else 1
                dp_crosses_nodes = dp > dp_per_node
                dp_bw = inter_node_bandwidth_gbps if dp_crosses_nodes else intra_node_bandwidth_gbps
                dp_msg_size = params_per_gpu * bytes_per_param
                dp_comm = (dp_msg_size * 2 * (dp - 1) / dp) / (dp_bw * 1e9 / 8) if dp > 1 else 0
                
                # PP communication
                pp_crosses_nodes = pp > gpus_per_node
                pp_bw = inter_node_bandwidth_gbps if pp_crosses_nodes else intra_node_bandwidth_gbps
                pp_msg_size = seq_len * dmodel * micro_batch * bytes_per_param
                pp_comm = (pp_msg_size * (pp - 1) * num_microbatches * 2) / (pp_bw * 1e9 / 8) if pp > 1 else 0
                
                # Pipeline bubble overhead
                bubble_overhead = (pp - 1) / (num_microbatches + pp - 1) if pp > 1 else 0
                
                # Penalize crossing nodes for TP heavily (latency sensitive)
                tp_latency_penalty = 1.0
                if tp > 1 and tp_crosses_nodes:
                    tp_latency_penalty = 3.0  # Heavy penalty for inter-node TP
                
                # Total estimated time
                score = (compute_time * (1 + bubble_overhead) + 
                         tp_comm * 1000 * tp_latency_penalty + 
                         dp_comm * 500 + 
                         pp_comm * 500)
                
                if score < best_score:
                    best_score = score
                    best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": micro_batch}
    
    if best_config is None:
        # Fallback: try aggressive memory saving with high PP and moderate TP
        for pp in [8, 4, 2, 1]:
            for tp in sorted(valid_tps, reverse=True):
                if tp * pp > total_gpus:
                    continue
                dp = total_gpus // (tp * pp)
                if dp < 1:
                    continue
                min_mb = math.ceil(batch_size / dp)
                if min_mb < 1:
                    min_mb = 1
                best_config = {"tp": tp, "dp": dp, "pp": pp, "micro_batch": min_mb}
                break
            if best_config:
                break
    
    if best_config is None:
        tp = valid_tps[-1] if valid_tps else 1
        best_config = {"tp": tp, "dp": 1, "pp": 1, "micro_batch": batch_size}
    
    return best_config
