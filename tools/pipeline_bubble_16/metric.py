from __future__ import annotations

from typing import Any

from tools.common.detect_profile_mode import detect_profile_mode
from tools.common.hta_wrappers import temporal_breakdown


MetricResult = dict[str, Any]


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    if profile_mode == "auto":
        profile_mode = detect_profile_mode(directory)

    if profile_mode != "torch":
        raise NotImplementedError("pipeline_bubble_16 requires torch/HTA traces")

    tb = temporal_breakdown(directory)
    bubble = {}
    for rank, entry in tb.items():
        idle = float(entry.get("idle_time", 0.0))
        compute = float(entry.get("compute_time", 0.0))
        noncompute = float(entry.get("non_compute_time", entry.get("noncompute_time", 0.0)))
        total = idle + compute + noncompute
        bubble[rank] = idle / total if total > 0 else 0.0

    avg = sum(bubble.values()) / len(bubble) if bubble else 0.0
    return {"bubble_fraction_per_rank": bubble, "avg_bubble_fraction": avg}
