"""Naive seed policy: pure data parallelism.

No model parallelism at all — just replicate the model on every GPU
and do data-parallel training. This is the simplest possible config
and a common starting point for someone who hasn't tuned parallelism.
"""


def generate_config(workload: dict, environment: dict) -> dict:
    total_gpus = environment.get("total_gpus", 16)

    return {
        "tp": 4,
        "dp": 4,
        "pp": 1,
        "micro_batch_size": 2,
        "activation_checkpointing": False,
    }