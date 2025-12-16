from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from tools.common.detect_profile_mode import detect_profile_mode
from tools.iter_time_16.metric import metric_cal as iter_time_metric


MetricResult = dict[str, Any]


def _load_per_rank_steps(trace_dir: str) -> dict[int, dict[int, float]]:
    path = Path(trace_dir) / "step_times_per_rank.json"
    if not path.exists():
        return {}
    with path.open() as fh:
        data = json.load(fh)
    result: dict[int, dict[int, float]] = {}
    if isinstance(data, dict):
        for rank, steps in data.items():
            if isinstance(steps, dict):
                result[int(rank)] = {int(k): float(v) for k, v in steps.items()}
    return result


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    if profile_mode == "auto":
        profile_mode = detect_profile_mode(directory)

    per_rank = _load_per_rank_steps(directory)
    if not per_rank:
        # Fallback: use global step time; straggler is zero.
        iter_stats = iter_time_metric(directory, profile_mode=profile_mode)
        return {
            "per_step_straggler_lag_sec": {},
            "avg_lag_sec": 0.0,
            "p95_lag_sec": 0.0,
            "iter_time": iter_stats,
        }

    lags = {}
    for step in {s for steps in per_rank.values() for s in steps}:
        durations = [per_rank[r].get(step, 0.0) for r in per_rank]
        lag = max(durations) - min(durations)
        lags[step] = lag

    values = list(lags.values())
    return {
        "per_step_straggler_lag_sec": {str(k): v for k, v in lags.items()},
        "avg_lag_sec": mean(values) if values else 0.0,
        "p95_lag_sec": sorted(values)[int(0.95 * len(values)) - 1] if values else 0.0,
    }
