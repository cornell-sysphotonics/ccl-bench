from __future__ import annotations

from typing import Any

from tools.common.detect_profile_mode import detect_profile_mode
from tools.common.hta_wrappers import comm_comp_overlap


MetricResult = int | float | dict[str, Any]


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    if profile_mode == "auto":
        profile_mode = detect_profile_mode(directory)

    if profile_mode == "torch":
        return comm_comp_overlap(directory)

    if profile_mode == "nsys":
        raise NotImplementedError("comm_comp_overlap_16 is only supported for torch traces")

    raise ValueError(f"Unknown profile_mode: {profile_mode}")
