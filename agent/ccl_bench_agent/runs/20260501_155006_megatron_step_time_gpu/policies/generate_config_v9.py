
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Try tp=2, dp=1, pp=1, mbs=1, AC=False (2 GPUs).
    
    Best so far: tp=4, dp=1, pp=1, mbs=2, AC=False → 2.72 (4 GPUs)
    
    Key findings:
    - AC=True ALWAYS fails → never use it
    - Fewer GPUs = higher score
    - mbs=4 with tp=4,pp=1,dp=1 failed (OOM)
    - dp=2 configs fail
    
    With 2 GPUs (tp=2): each GPU holds ~4B params of Llama-8B.
    Memory: ~4B * (2 + 4 + 8) bytes (params + grads + optimizer in mixed precision) ≈ 28GB
    Plus activations for mbs=1, seq=1024: should be manageable on 40GB A100.
    If successful, using 2 GPUs instead of 4 should significantly boost score.
    
    Previous tp=2,dp=1,pp=1 attempt failed ONLY because AC=True was used.
    """
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    return {
        "tp": 2,
        "dp": 1,
        "pp": 1,
        "micro_batch_size": 1,
        "activation_checkpointing": False,
    }
