#!/usr/bin/env python3
"""Parse the TorchTitan CLI and enforce the matched WL5 configuration."""

from torchtitan.config import ConfigManager


config = ConfigManager().parse_args()

expected = {
    "training.local_batch_size": (config.training.local_batch_size, 4),
    "training.seq_len": (config.training.seq_len, 1024),
    "training.steps": (config.training.steps, 6),
    "training.dtype": (config.training.dtype, "bfloat16"),
    "parallelism.data_parallel_replicate_degree": (
        config.parallelism.data_parallel_replicate_degree,
        1,
    ),
    "parallelism.data_parallel_shard_degree": (
        config.parallelism.data_parallel_shard_degree,
        2,
    ),
    "parallelism.tensor_parallel_degree": (
        config.parallelism.tensor_parallel_degree,
        4,
    ),
    "parallelism.pipeline_parallel_degree": (
        config.parallelism.pipeline_parallel_degree,
        1,
    ),
    "parallelism.expert_parallel_degree": (
        config.parallelism.expert_parallel_degree,
        8,
    ),
    "optimizer.lr": (config.optimizer.lr, 8e-4),
    "lr_scheduler.warmup_steps": (config.lr_scheduler.warmup_steps, 2),
    "dataloader.dataset": (config.dataloader.dataset, "c4_test"),
    "profiling.profile_freq": (config.profiling.profile_freq, 4),
    "profiling.enable_profiling": (config.profiling.enable_profiling, True),
    "checkpoint.enable": (config.checkpoint.enable, False),
    "activation_checkpoint.mode": (config.activation_checkpoint.mode, "none"),
    "compile.enable": (config.compile.enable, False),
}

wrong = {
    name: {"actual": actual, "expected": wanted}
    for name, (actual, wanted) in expected.items()
    if actual != wanted
}
if wrong:
    raise RuntimeError(f"Unmatched TorchTitan configuration: {wrong}")

print("TorchTitan configuration preflight passed")
