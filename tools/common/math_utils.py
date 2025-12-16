"""Shared math helpers for metrics."""

from __future__ import annotations


def compute_mfu(
    tokens_per_sec: float, flops_per_token: float, num_gpus: int, peak_flops_per_gpu_tflops: float
) -> float:
    actual_flops = tokens_per_sec * flops_per_token
    peak_flops = num_gpus * peak_flops_per_gpu_tflops * 1e12
    return actual_flops / peak_flops if peak_flops > 0 else 0.0


def tokens_per_second_per_gb(tokens_per_sec: float, used_bytes: int) -> float:
    used_gb = used_bytes / (1024**3)
    return tokens_per_sec / used_gb if used_gb > 0 else 0.0


def tokens_per_step_per_gb(tokens_per_step: float, used_bytes: int) -> float:
    used_gb = used_bytes / (1024**3)
    return tokens_per_step / used_gb if used_gb > 0 else 0.0
