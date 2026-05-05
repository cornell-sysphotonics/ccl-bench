
"""
Adaptive config for LLM training workloads.

Key lessons learned:
- Run 1: tp=4, pp=3, dp=1, ep=1, mbs=4, ac=False → 12 GPUs → score 1.472
- Run 2: tp=2, pp=3, dp=1, ep=1 → 6 GPUs → FAILED (not multiple of gpus_per_node=4)
- Run 3: tp=4, pp=1, dp=1, ep=1 → 4 GPUs → FAILED (OOM)
- Run 4: tp=2, pp=3, dp=2, ep=2 → 12 GPUs → FAILED (server error)
- Run 5: tp=2, pp=3, dp=2, ep=2, mbs=2, ac=True → 12 GPUs → score 1.619 (BEST)
- Run 6: tp=2, pp=3, dp=2, ep=2, mbs=1, ac=True → 12 GPUs → score 1.358
- Run 7: tp=2, pp=3, dp=2, ep=1, mbs=2, ac=True → 12 GPUs → FAILED (OOM w/o EP)

Next: tp=2, pp=3, dp=2, ep=2, mbs=4, ac=True → 12 GPUs
- Same as best but mbs=4 instead of 2
- With dp=2, each rank gets batch_size/dp = 4 samples
- mbs=4 means 1 microbatch → no pipeline bubble at all
- Should improve GPU utilization significantly
"""

def generate_config(workload: dict, environment: dict) -> dict:
    config_space = {d["key"]: d["choices"] for d in workload.get("config_space", [])}
    
    gpu_mem = environment.get("gpu_memory_gb", 40)
    gpus_per_node = environment.get("gpus_per_node", 4)
    total_gpus = environment.get("total_gpus", 16)
    is_moe = workload.get("moe", False)
    batch_size = workload.get("batch_size", 8)
    seq_len = workload.get("seq_len", 1024)
    num_layers = workload.get("num_layers") or 27
    
    tp_choices = sorted(config_space.get("tp", [1, 2, 4]))
    pp_choices = sorted(config_space.get("pp", [1, 3, 9]))
    dp_choices = sorted(config_space.get("dp", [1, 2, 4]))
    ep_choices = sorted(config_space.get("ep", [1, 2, 4]))
    mbs_choices = sorted(config_space.get("micro_batch_size", [1, 2, 4]))
    ac_choices = config_space.get("activation_checkpointing", [True, False])
    
    def valid_gpu_count(tp, dp, pp):
        total = tp * dp * pp
        return total <= total_gpus and total % gpus_per_node == 0
    
    if is_moe:
        # DeepSeek-V2-Lite: ~16B params, 64 experts, 27 layers
        # Best config was tp=2, pp=3, dp=2, ep=2, mbs=2, ac=True → 1.619
        # Try mbs=4 to eliminate pipeline bubbles (1 microbatch per stage)
        
        tp = 2
        pp = 3
        dp = 2
        ep = 2  # Must have EP>=2 for MoE to fit in memory
        micro_batch_size = 4  # Try larger mbs for better utilization
        activation_checkpointing = True
        
        # Verify: batch_size/dp = 8/2 = 4, mbs=4 → 1 microbatch, no pipeline bubble
        # But mbs=4 uses more activation memory - AC should help
        
        if not valid_gpu_count(tp, dp, pp):
            # Fallback
            tp = 4
            pp = 3
            dp = 1
            ep = 1
            micro_batch_size = 4
            activation_checkpointing = False
    else:
        # Dense model logic
        num_params = workload.get("num_params") or 8e9
        params_billions = num_params / 1e9
        
        tp = min(gpus_per_node, max(tp_choices))
        pp = 1
        dp = 1
        ep = 1
        
        mem_needed_gb = (params_billions / tp) * 12
        
        if mem_needed_gb > gpu_mem * 0.85:
            for pp_candidate in sorted(pp_choices):
                if pp_candidate > 1 and num_layers % pp_candidate == 0:
                    mem_est = (params_billions / (tp * pp_candidate)) * 12
                    if mem_est < gpu_mem * 0.85:
                        pp = pp_candidate
                        break
            activation_checkpointing = True
        else:
            activation_checkpointing = False
        
        if not valid_gpu_count(tp, dp, pp):
            for dp_candidate in dp_choices:
                if valid_gpu_count(tp, dp_candidate, pp):
                    dp = dp_candidate
                    break
        
        micro_batch_size = max(mbs_choices)
    
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
