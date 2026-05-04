
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Analysis of history:
    - Best: tp=4, dp=1, pp=1, mbs=1, ckpt=True → 4 GPUs, score=2.736
    - tp=4, dp=1, pp=1, mbs=2, ckpt=True → 4 GPUs, score=2.697
    - tp=4, dp=1, pp=1, mbs=4, ckpt=True → FAILED (OOM)
    - tp=4 is required; fewer GPUs = better score
    
    Try: tp=4, dp=1, pp=1, mbs=1, ckpt=False
    - Without activation checkpointing, forward pass is faster (no recomputation)
    - mbs=1 keeps activation memory small
    - With tp=4 sharding the model, memory per GPU should be manageable:
      ~4GB weights + ~12GB optimizer states + small activations for mbs=1
    - If it fits in memory, should get better throughput → higher score
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
    config["micro_batch_size"] = 1
    config["activation_checkpointing"] = False
    
    return config
