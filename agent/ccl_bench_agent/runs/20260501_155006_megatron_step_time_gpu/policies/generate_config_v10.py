
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Try tp=1, dp=1, pp=4, mbs=4, AC=False (4 GPUs).
    
    Best so far: tp=4, dp=1, pp=1, mbs=2, AC=False → 2.72 (4 GPUs)
    
    Key findings:
    - AC=True ALWAYS fails → never use it
    - Fewer GPUs = higher score
    - 2 GPUs OOM even with mbs=1 (tp=2,pp=1 or tp=1,pp=2 style)
    - dp=2 fails
    - tp=4,pp=1,mbs=2 (4 GPUs) → 2.72 (best)
    - tp=2,pp=2,mbs=2 (4 GPUs) → 2.561
    
    Now try pp=4 with tp=1 (4 GPUs). PP=4 splits model into 4 stages,
    each GPU holds ~2B params. Memory per GPU much lower.
    With mbs=4, pipeline can be more efficient with more micro-batches.
    """
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    return {
        "tp": 1,
        "dp": 1,
        "pp": 4,
        "micro_batch_size": 4,
        "activation_checkpointing": False,
    }
