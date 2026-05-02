
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Optimize within 4 GPUs (tp=2, pp=2, dp=1) which is known to work.
    
    History:
    - 8 GPUs (tp=4, dp=1, pp=2), mbs=1, no AC → 1.517
    - 2 GPUs (tp=2, dp=1, pp=1), mbs=1, AC=True → FAILED
    - 4 GPUs (tp=2, dp=1, pp=2), mbs=2, no AC → 2.561 (BEST)
    - 2 GPUs (tp=1, dp=1, pp=2), mbs=4, AC=True → FAILED
    
    Try increasing micro_batch_size to 4 for better compute efficiency.
    Keep activation_checkpointing=False since it worked before.
    """
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    return {
        "tp": 2,
        "dp": 1,
        "pp": 2,
        "micro_batch_size": 4,
        "activation_checkpointing": False,
    }
