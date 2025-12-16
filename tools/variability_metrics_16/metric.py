from __future__ import annotations

from math import sqrt
from statistics import mean
from typing import Any

from tools.common.detect_profile_mode import detect_profile_mode
from tools.iter_time_16.metric import metric_cal as iter_time_metric
from tools.throughput_tokens_16.metric import metric_cal as throughput_metric


MetricResult = dict[str, Any]


def _coeff_var(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    if mu == 0:
        return 0.0
    variance = sum((v - mu) ** 2 for v in values) / len(values)
    return sqrt(variance) / mu


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    if profile_mode == "auto":
        profile_mode = detect_profile_mode(directory)

    iter_stats = iter_time_metric(directory, profile_mode=profile_mode)
    durations = list(iter_stats.get("per_step_time_sec", {}).values())
    throughput = throughput_metric(directory, profile_mode=profile_mode)

    return {
        "step_time_cv": _coeff_var([float(v) for v in durations]),
        "throughput_tokens_per_sec": throughput.get("throughput_tokens_per_sec"),
    }
