from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
from typing import Any, cast

from tools.common.detect_profile_mode import detect_profile_mode


MetricResult = dict[str, Any]
_LOGGER = logging.getLogger(__name__)


def _parse_assignments_file(path: Path) -> dict[Any, Any] | None:
    try:
        return cast("dict[Any, Any]", json.loads(path.read_text()))
    except Exception as exc:
        _LOGGER.warning("Failed to parse assignment log %s: %s", path, exc)
        return None


def _load_assignments(trace_dir: str) -> list[dict]:
    paths = list(Path(trace_dir).glob("moe_assignments*.json"))
    data: list[dict] = []
    for path in paths:
        parsed = _parse_assignments_file(path)
        if parsed is not None:
            data.append(parsed)
    return data


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    if profile_mode == "auto":
        profile_mode = detect_profile_mode(directory)
    _ = profile_mode  # unused but retained for symmetry

    logs = _load_assignments(directory)
    if not logs:
        return {"error": "No MoE assignment logs found (moe_assignments*.json)"}

    global_counts: Counter[str] = Counter()
    per_rank: dict[str, Counter[str]] = {}

    for entry in logs:
        rank = str(entry.get("rank", "unknown"))
        counts = entry.get("token_counts")
        if counts is None:
            continue
        local_counter = Counter({str(i): int(c) for i, c in enumerate(counts)})
        per_rank.setdefault(rank, Counter()).update(local_counter)
        global_counts.update(local_counter)

    total_tokens = sum(global_counts.values()) or 1
    imbalance = (
        (max(global_counts.values()) / min(global_counts.values())) if global_counts else 0.0
    )

    return {
        "global_expert_tokens": dict(global_counts),
        "per_rank_expert_tokens": {r: dict(c) for r, c in per_rank.items()},
        "imbalance_ratio": imbalance,
        "total_tokens": total_tokens,
    }
