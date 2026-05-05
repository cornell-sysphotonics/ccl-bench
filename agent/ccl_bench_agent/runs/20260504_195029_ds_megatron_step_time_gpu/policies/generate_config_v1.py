"""
Adaptive config for DeepSeek-V2-Lite MoE model.
Strategy: Minimize GPU count to maximize gpu_step_score while fitting in memory.
Try TP=2, PP=3, DP=1, EP=1 = 6 GPUs with activation checkpointing.
"""

def generate_config(workload: dict, environment: dict) -> dict:
    config_space = {d["key"]: d["choices"] for d in workload.get("config_space", [])}
    
    gpu_mem = environment.get("gpu_memory_gb", 40)
    gpus_per_node = environment.get("gpus_per_node", 4)
    total_gpus = environment.get("total_gpus", 16)
    is_moe = workload.get("moe", False)
    batch_size = workload.get("batch_size", 8)
    seq_len = workload.get("seq_len", 1024)
    
    # For DeepSeek-V2-Lite MoE on A100-40GB:
    # - 27 layers, 64 experts, ~16B params total but sparse
    # - Try to minimize GPU count for better score
    # - Use activation checkpointing to reduce memory pressure
    
    # Strategy: TP=2 keeps tensor parallel within node
    # PP=3 divides 27 layers into 9 layers per stage
    # DP=1 minimal data parallelism
    # EP=1 no expert parallelism (avoids alltoall)
    # Total GPUs = 2*1*3 = 6
    
    tp = 2
    pp = 3
    dp = 1
    ep = 1
    micro_batch_size = 2
    activation_checkpointing = True
    
    # Validate choices against config space
    if tp not in config_space.get("tp", [1, 2, 4]):
        tp = 4
    if pp not in config_space.get("pp", [1, 3, 9]):
        pp = 3
    if dp not in config_space.get("dp", [1, 2, 4]):
        dp = 1
    if ep not in config_space.get("ep", [1, 2, 4]):
        ep = 1
    if micro_batch_size not in config_space.get("micro_batch_size", [1, 2, 4]):
        micro_batch_size = 1
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "ep": ep,
        "micro_batch_size": micro_batch_size,
        "activation_checkpointing": activation_checkpointing,
    }
