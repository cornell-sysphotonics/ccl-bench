
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Best config so far: tp=4, dp=1, pp=1, mbs=2, activation_checkpointing=False → 2.76
    
    Try mbs=4 without activation checkpointing to potentially improve throughput.
    Larger micro-batch can improve GPU compute utilization.
    
    Memory estimate with tp=4, dp=1, pp=1, mbs=4, no checkpointing:
    - Model weights (bf16): ~16GB / 4 = ~4GB per GPU
    - Optimizer states: ~24GB / 4 = ~6GB per GPU  
    - Activations for mbs=4, seq=1024: moderate, but no checkpointing means full storage
    - Gradients: ~4GB per GPU
    - Total estimate: ~20-30GB, should fit in 40GB
    """
    config = {}
    
    # Parse config space
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    config["tp"] = 4
    config["dp"] = 1
    config["pp"] = 1
    config["micro_batch_size"] = 4
    config["activation_checkpointing"] = False
    
    return config
