from __future__ import annotations

from collections import Counter
from typing import Any

from tools.common.detect_profile_mode import detect_profile_mode
from tools.common.hta_wrappers import comm_stats
from tools.common.nsys_stats import run_nsys_stats_in_dir


MetricResult = int | float | dict[str, Any]


def _aggregate_htc(stats: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for entry in stats.values():
        for name, data in entry.items():
            total_calls = data["count"] if isinstance(data, dict) and "count" in data else 0
            counts[name] += total_calls
    return dict(counts)


def _aggregate_nsys(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        name = row.get("Name") or row.get("API Name") or ""
        if "nccl" in name.lower():
            try:
                calls_str = row.get("Num Calls") or row.get("Calls") or "0"
                calls = int(calls_str)
            except ValueError:
                calls = 0
            counts[name] += calls
    return dict(counts)


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    if profile_mode == "auto":
        profile_mode = detect_profile_mode(directory)

    if profile_mode == "torch":
        stats = comm_stats(directory)
        return _aggregate_htc(stats)

    if profile_mode == "nsys":
        tables = run_nsys_stats_in_dir(directory, reports=("cuda_api_sum",))
        rows = tables.get("cuda_api_sum", [])
        return _aggregate_nsys(rows)

    raise ValueError(f"Unknown profile_mode: {profile_mode}")
