
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Try tp=2, dp=1, pp=1 (2 GPUs) with mbs=1 and activation_checkpointing=True.
    
    If this works, 2 GPUs should give a much higher score than 4 GPUs.
    - Llama-3.1-8B bf16 weights: ~16GB, split by tp=2 → ~8GB/GPU
    - Optimizer states (Adam): ~3x weights in mixed precision → ~24GB total, ~12GB/GPU
    - Activations with mbs=1, seq=1024, checkpointing: relatively small
    - Total estimate: ~22-28GB per GPU, should fit in 40GB A100
    
    Previous tp=2 failure was with dp=2, mbs=4 (much more memory pressure).
    """
    config = {}
    
    # Parse config space
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    config["tp"] = 2
    config["dp"] = 1
    config["pp"] = 1
    config["micro_batch_size"] = 1
    config["activation_checkpointing"] = True
    
    return config
