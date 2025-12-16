from __future__ import annotations

from typing import Any

from tools.common.detect_profile_mode import detect_profile_mode
from tools.common.metadata import load_run_metadata, tokens_per_step
from tools.iter_time_16.metric import metric_cal as iter_time_metric


MetricResult = dict[str, Any]


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    if profile_mode == "auto":
        profile_mode = detect_profile_mode(directory)

    meta = load_run_metadata(directory)
    tp_step = tokens_per_step(meta)

    iter_stats = iter_time_metric(directory, profile_mode=profile_mode)
    avg_step = iter_stats.get("avg_step_time_sec", 0.0) or 0.0
    throughput = tp_step / avg_step if avg_step else 0.0

    return {
        "tokens_per_step": tp_step,
        "avg_step_time_sec": avg_step,
        "throughput_tokens_per_sec": throughput,
        "num_steps_profiled": iter_stats.get("num_steps", 0),
    }
