from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

from tools.common.detect_profile_mode import detect_profile_mode


MetricResult = dict[str, Any]
_LOGGER = logging.getLogger(__name__)

_DEF_FILES = ("training_metrics.json", "training_log.json")


def _parse_metrics_file(path: Path) -> dict[str, Any] | None:
    try:
        return cast("dict[str, Any]", json.loads(path.read_text()))
    except Exception as exc:
        _LOGGER.warning("Failed to parse training metrics from %s: %s", path, exc)
        return None


def _load_training_metrics(trace_dir: str) -> dict[str, Any]:
    for name in _DEF_FILES:
        path = Path(trace_dir) / name
        if not path.exists():
            continue
        metrics = _parse_metrics_file(path)
        if metrics is not None:
            return metrics
    return {}


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    if profile_mode == "auto":
        profile_mode = detect_profile_mode(directory)
    _ = profile_mode

    metrics = _load_training_metrics(directory)
    return metrics if metrics else {"info": "No training quality metrics found"}
