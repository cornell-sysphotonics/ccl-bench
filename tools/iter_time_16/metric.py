from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from tools.common.detect_profile_mode import detect_profile_mode
from hta.trace_analysis import TraceAnalysis


MetricResult = dict[str, Any]


_STEP_FILES = ("step_times.json", "step_durations.json")


def _load_step_times_from_file(trace_dir: str) -> dict[int, float]:
    base = Path(trace_dir)
    for name in _STEP_FILES:
        path = base / name
        if not path.exists():
            continue
        with path.open() as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return {idx: float(val) for idx, val in enumerate(data)}
        if isinstance(data, dict):
            if "steps" in data and isinstance(data["steps"], list):
                return {
                    int(item.get("step", idx)): float(
                        item.get("duration_sec", item.get("duration", 0.0))
                    )
                    for idx, item in enumerate(data["steps"])
                }
            if all(isinstance(v, (int, float)) for v in data.values()):
                return {int(k): float(v) for k, v in data.items()}
    return {}


def _step_times_from_hta(trace_dir: str) -> dict[int, float]:
    trace = TraceAnalysis(trace_dir=trace_dir)
    for candidate in ("get_step_time", "get_step_time_series", "get_step_time_df"):
        if hasattr(trace, candidate):
            func = getattr(trace, candidate)
            result = func()
            if hasattr(result, "to_dict"):
                result = result.to_dict()
            if isinstance(result, dict):
                # assume dict[step]->duration
                return {int(k): float(v) for k, v in result.items()}
    raise NotImplementedError("HTA does not expose step timing helper; provide step_times.json")


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    if profile_mode == "auto":
        profile_mode = detect_profile_mode(directory)

    step_times = _load_step_times_from_file(directory)

    if not step_times and profile_mode == "torch":
        try:
            step_times = _step_times_from_hta(directory)
        except Exception:
            step_times = {}

    if not step_times:
        raise RuntimeError(
            "No step timing information found; provide step_times.json or ensure HTA exposes step timing"
        )

    durations = list(step_times.values())
    return {
        "per_step_time_sec": {str(k): v for k, v in step_times.items()},
        "avg_step_time_sec": mean(durations),
        "p95_step_time_sec": sorted(durations)[int(0.95 * len(durations)) - 1]
        if durations
        else None,
        "num_steps": len(durations),
    }
