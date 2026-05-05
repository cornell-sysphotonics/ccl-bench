
"""
Adaptive config for LLM training workloads.

Key lessons from 12 runs:
- Best: tp=2, pp=3, dp=2, ep=2, mbs=4, ac=False → 12 GPUs → 1.934
- Only 12 GPUs works (pp=3 required, must be multiple of gpus_per_node=4)
- pp=1 always OOMs, pp=9 very slow, pp=3 is optimal
- ep=2 much better than ep=1 for MoE
- mbs=4 >> mbs=2 >> mbs=1
- ac=False slightly better than ac=True
- 6 GPU configs fail (not multiple of 4)

Next exploration: tp=1, pp=3, dp=4, ep=4, mbs=4, ac=True → 12 GPUs
- tp=1 eliminates tensor parallel communication overhead
- dp=4 with ep=4 spreads experts across more GPUs
- ac=True to handle memory with tp=1
- Risk: tp=1 might OOM even with ac=True, but worth trying
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
        # Try tp=1, pp=3, dp=4, ep=4 → 12 GPUs
        # This eliminates tensor parallel overhead, uses more expert parallelism
        
        tp = 1
        pp = 3
        dp = 4
        ep = 4  # ep divides dp=4 ✓, ep divides 64 experts ✓
        micro_batch_size = 4
        activation_checkpointing = True  # Need AC since tp=1 puts more params per GPU
        
        # Validate GPU count: 1*4*3 = 12, 12 % 4 == 0 ✓, 12 <= 16 ✓
        if not valid_gpu_count(tp, dp, pp):
            # Fallback to proven best
            tp = 2
            pp = 3
            dp = 2
            ep = 2
            micro_batch_size = 4
            activation_checkpointing = False
        
        # Validate EP divides dp
        if dp % ep != 0:
            ep = min(e for e in ep_choices if dp % e == 0)
    else:
        # Dense model logic
        num_params = workload.get("num_params") or 8e9
        params_billions = num_params / 1e9
        
        tp = min(gpus_per_node, max(tp_choices))
        pp = 1
        dp = 1
        ep = 1 if 1 in ep_choices else ep_choices[0]
        
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
        micro_batch_size = mbs_choices[-1]
    if activation_checkpointing not in ac_choices:
        activation_checkpointing = True
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "ep": ep,
        "micro_batch_size": micro_batch_size,
        "activation_checkpointing": activation_checkpointing,
    }
