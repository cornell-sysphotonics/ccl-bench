from __future__ import annotations

from collections import defaultdict
from typing import Any

from tools.common.detect_profile_mode import detect_profile_mode
from tools.common.hta_wrappers import comm_stats
from tools.common.nsys_stats import run_nsys_stats_in_dir


MetricResult = dict[str, Any]


def _bytes_from_htc(stats: dict[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for entry in stats.values():
        for name, data in entry.items():
            if isinstance(data, dict):
                bytes_val = data.get("total_bytes") or data.get("bytes", 0)
                totals[name] += float(bytes_val)
    totals["total_bytes"] = sum(totals.values())
    return totals


def _bytes_from_nsys(rows: list[dict[str, str]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        name = row.get("Name") or row.get("API Name") or ""
        if "nccl" not in name.lower():
            continue
        size_fields = ["Average", "Total", "Size"]
        size = 0.0
        for key in size_fields:
            val = row.get(key)
            if val:
                try:
                    size = float(val)
                    break
                except ValueError:
                    continue
        totals[name] += size
    totals["total_bytes"] = sum(totals.values())
    return totals


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    if profile_mode == "auto":
        profile_mode = detect_profile_mode(directory)

    if profile_mode == "torch":
        stats = comm_stats(directory)
        return _bytes_from_htc(stats)

    if profile_mode == "nsys":
        tables = run_nsys_stats_in_dir(directory, reports=("cuda_api",))
        rows = tables.get("cuda_api", [])
        return _bytes_from_nsys(rows)

    raise ValueError(f"Unknown profile_mode: {profile_mode}")
