
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Try tp=4, dp=1, pp=1, mbs=4, AC=False.
    
    Best so far: tp=4, dp=1, pp=1, mbs=2, AC=False → 2.72
    
    Key insight: AC=True ALWAYS fails (3/3 attempts), so never use it.
    
    Increasing mbs from 2 to 4 could improve GPU compute efficiency/utilization
    since larger batches better utilize tensor cores. With tp=4 on 4 GPUs within
    one node (600 Gbps NVLink), each GPU holds ~2B params. Memory per GPU:
    ~2B * 12 bytes (param + grad + optimizer) = ~24GB + activations.
    With mbs=4 and seq=1024, activations might fit in remaining ~16GB.
    
    Without activation checkpointing this time (AC always fails).
    """
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    return {
        "tp": 4,
        "dp": 1,
        "pp": 1,
        "micro_batch_size": 4,
        "activation_checkpointing": False,
    }
