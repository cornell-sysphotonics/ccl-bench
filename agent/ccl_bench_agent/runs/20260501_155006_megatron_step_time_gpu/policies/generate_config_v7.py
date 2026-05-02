
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Build on the best config (tp=4, dp=1, pp=1, mbs=2, AC=False → 2.72).
    
    Try tp=4, dp=1, pp=1 (4 GPUs), mbs=4, AC=True.
    - Activation checkpointing should save enough memory to allow mbs=4
    - Larger micro-batch improves compute efficiency / GPU utilization
    - 4 GPUs with tp=4 within single node uses fast NVLink
    - pp=1 avoids pipeline bubbles
    
    If this works with higher throughput, the score should improve.
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
        "activation_checkpointing": True,
    }
