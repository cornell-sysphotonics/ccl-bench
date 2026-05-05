
"""
Adaptive config for LLM training workloads.

Key lessons learned:
- Run 1: tp=4, pp=3, dp=1, ep=1, mbs=4, ac=False → 12 GPUs → score 1.472
- Run 5: tp=2, pp=3, dp=2, ep=2, mbs=2, ac=True → 12 GPUs → score 1.619
- Run 6: tp=2, pp=3, dp=2, ep=2, mbs=1, ac=True → 12 GPUs → score 1.358
- Run 7: tp=2, pp=3, dp=2, ep=1, mbs=2, ac=True → 12 GPUs → OOM (ep=1 with dp=2 OOMs)
- Run 8: tp=2, pp=3, dp=2, ep=2, mbs=4, ac=True → 12 GPUs → score 1.917 (BEST!)

Pattern: mbs=4 >> mbs=2 >> mbs=1. Higher micro_batch_size improves GPU utilization.
EP>=2 is needed for MoE to fit in memory with dp>=2.

Next: Try tp=4, pp=1, dp=4, ep=4, mbs=2, ac=True → 16 GPUs.
- No pipeline parallelism → no pipeline bubbles
- tp=4 within node for memory reduction
- ep=4 distributes 64 experts across 4 groups (16 experts/GPU group)
- dp=4, batch/dp=2, mbs=2 → 1 microbatch
- Previous OOM with tp=4,pp=1 was without EP; ep=4 should fix memory
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
        # Try pp=1 to eliminate pipeline bubbles entirely
        # Use tp=4, dp=4, ep=4 for maximum parallelism
        
        tp = 4
        pp = 1
        dp = 4
        ep = 4  # Distribute experts widely to fit in memory
        micro_batch_size = 2  # batch/dp = 8/4 = 2, mbs=2 → 1 microbatch
        activation_checkpointing = True
        
        if not valid_gpu_count(tp, dp, pp):
            # Fallback to proven best config
            tp = 2
            pp = 3
            dp = 2
            ep = 2
            micro_batch_size = 4
            activation_checkpointing = True
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
