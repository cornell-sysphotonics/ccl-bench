"""
Adaptive config for LLM training workloads.
Key lessons learned:
- Total GPUs (tp*dp*pp) must align with node topology (multiples of gpus_per_node preferred)
- Run 1 (tp=4,pp=3,dp=1,ep=1,mbs=4,ac=False) = 12 GPUs → score 1.472
- Run 2 (tp=2,pp=3,dp=1,ep=1,mbs=2,ac=True) = 6 GPUs → FAILED (cross-node rank issue)
- Try 4 GPUs (single node) to maximize per-GPU efficiency
"""

def generate_config(workload: dict, environment: dict) -> dict:
    config_space = {d["key"]: d["choices"] for d in workload.get("config_space", [])}
    
    gpu_mem = environment.get("gpu_memory_gb", 40)
    gpus_per_node = environment.get("gpus_per_node", 4)
    total_gpus = environment.get("total_gpus", 16)
    is_moe = workload.get("moe", False)
    batch_size = workload.get("batch_size", 8)
    seq_len = workload.get("seq_len", 1024)
    num_layers = workload.get("num_layers") or 27  # DeepSeek-V2-Lite default
    
    # Parse valid choices
    tp_choices = sorted(config_space.get("tp", [1, 2, 4]))
    pp_choices = sorted(config_space.get("pp", [1, 3, 9]))
    dp_choices = sorted(config_space.get("dp", [1, 2, 4]))
    ep_choices = sorted(config_space.get("ep", [1, 2, 4]))
    mbs_choices = sorted(config_space.get("micro_batch_size", [1, 2, 4]))
    ac_choices = config_space.get("activation_checkpointing", [True, False])
    
    # Helper: ensure total GPUs is a valid multiple of gpus_per_node
    def valid_gpu_count(tp, dp, pp):
        total = tp * dp * pp
        return total <= total_gpus and (total % gpus_per_node == 0 or total <= gpus_per_node)
    
    # Strategy: maximize gpu_step_score
    # gpu_step_score likely rewards faster step time and/or fewer GPUs
    # Try to use minimal GPUs while fitting model in memory
    
    if is_moe:
        # MoE model (e.g., DeepSeek-V2-Lite: ~16B params, 64 experts, 27 layers)
        # With 64 experts, each expert is relatively small
        # Sparse activation means effective params per forward pass is much smaller
        
        # Strategy: Try single-node (4 GPUs) first
        # TP=4, PP=1, DP=1, EP=1 = 4 GPUs
        # Memory: ~16B params / 4 (TP) = 4B params/GPU
        # But MoE has all 64 experts stored, only ~6 active
        # Training memory ~ 4B * 12 bytes = 48GB > 40GB → need AC or PP
        
        # Try TP=4, PP=3, DP=1, EP=1 = 12 GPUs (known to work from run 1)
        # But let's try TP=4, PP=1, DP=1, EP=1 = 4 GPUs with AC=True, MBS=1
        # If that's too much memory, fall back to PP=3
        
        # Attempt: minimize GPUs for better efficiency
        # Option A: TP=4, PP=1, DP=1, EP=1 = 4 GPUs (single node, fast comm)
        tp = 4
        pp = 1
        dp = 1
        ep = 1
        micro_batch_size = 1
        activation_checkpointing = True
        
        # Verify this is valid
        if not valid_gpu_count(tp, dp, pp):
            # Fallback to known-good config
            tp = 4
            pp = 3
            dp = 1
            ep = 1
            micro_batch_size = 4
            activation_checkpointing = False
    else:
        # Dense model
        # Estimate memory needs
        num_params = workload.get("num_params") or 8e9  # default 8B
        params_per_gpu_billions = num_params / 1e9
        
        # Start with max TP within node
        tp = min(gpus_per_node, max(tp_choices))
        pp = 1
        dp = 1
        
        # Check if model fits: rough estimate params/tp * 12 bytes for training
        mem_needed_gb = (params_per_gpu_billions / tp) * 12
        
        if mem_needed_gb > gpu_mem * 0.85:
            # Need more parallelism - add PP
            for pp_candidate in sorted(pp_choices):
                if pp_candidate > 1 and num_layers % pp_candidate == 0:
                    mem_est = (params_per_gpu_billions / (tp * pp_candidate)) * 12
                    if mem_est < gpu_mem * 0.85:
                        pp = pp_candidate
                        break
            activation_checkpointing = True
        else:
            activation_checkpointing = False
        
        # Set micro batch size
        micro_batch_size = max(mbs_choices)
        ep = 1
        
        if not valid_gpu_count(tp, dp, pp):
            dp = 1
            pp = pp_choices[0]
    
    # Final validation against config space
    if tp not in tp_choices:
        tp = tp_choices[-1]
    if pp not in pp_choices:
        pp = pp_choices[0]
    if dp not in dp_choices:
        dp = dp_choices[0]
    if ep not in ep_choices:
        ep = ep_choices[0]
    if micro_batch_size not in mbs_choices:
        micro_batch_size = mbs_choices[0]
    if activation_checkpointing not in ac_choices:
        activation_checkpointing = True
    
    # EP constraint: must divide dp
    if is_moe and ep > 1 and dp % ep != 0:
        ep = 1
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "ep": ep,
        "micro_batch_size": micro_batch_size,
        "activation_checkpointing": activation_checkpointing,
    }
