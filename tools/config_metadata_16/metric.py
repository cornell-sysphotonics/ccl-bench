from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tools.common.detect_profile_mode import detect_profile_mode
from tools.common.metadata import load_run_metadata


MetricResult = dict[str, Any]

_LOGGER = logging.getLogger(__name__)


def _read_config_file(path: Path) -> str | None:
    try:
        return path.read_text()
    except Exception as exc:
        _LOGGER.warning("Failed to read config %s: %s", path, exc)
        return None


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    # profile_mode retained for symmetry but not used
    if profile_mode == "auto":
        profile_mode = detect_profile_mode(directory)
    _ = profile_mode

    meta = load_run_metadata(directory)

    toml_files = list(Path(directory).glob("*.toml"))
    configs = {}
    for path in toml_files:
        contents = _read_config_file(path)
        if contents is not None:
            configs[path.name] = contents

    return {"run_metadata": meta, "raw_configs": configs}
