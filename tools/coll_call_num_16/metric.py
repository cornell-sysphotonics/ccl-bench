from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from tools.common.detect_profile_mode import detect_profile_mode, list_torch_trace_dirs
from tools.common.nsys_stats import run_nsys_stats_in_dir


MetricResult = int | float | dict[str, Any]


def _aggregate_nsys(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        name = row.get("Name") or row.get("API Name") or ""
        if "nccl" in name.lower():
            try:
                calls_str = row.get("Num Calls") or row.get("Calls") or row.get("Instances") or "0"
                calls = int(calls_str)
            except ValueError:
                calls = 0
            counts[name] += calls
    return dict(counts)


def _aggregate_traces(trace_dir: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    trace_path = Path(trace_dir)
    trace_files = list(trace_path.glob("*trace.json"))
    if not trace_files:
        legacy = trace_path / "kineto_trace_0.json"
        trace_files = [legacy] if legacy.exists() else []

    for tf in trace_files:
        data = _safe_load_json(tf)
        if data is None:
            continue

        for event in data.get("traceEvents", []):
            name = event.get("name", "")
            lower = name.lower()
            if "nccl" in lower:
                counts[name] += 1

    return dict(counts)


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    if profile_mode == "auto":
        profile_mode = detect_profile_mode(directory)

    if profile_mode == "torch":
        trace_dirs = list_torch_trace_dirs(directory)
        if not trace_dirs:
            return {"error": "No torch trace JSON files found (expected *trace.json)"}
        # Aggregate across all iterations
        aggregated: dict[str, int] = {}
        for trace_dir in trace_dirs:
            partial = _aggregate_traces(str(trace_dir))
            for k, v in partial.items():
                aggregated[k] = aggregated.get(k, 0) + v
        return aggregated

    if profile_mode == "nsys":
        # Try summed CUDA API first; fall back to GPU kernel summary if API is missing
        tables = run_nsys_stats_in_dir(directory, reports=("cuda_api_sum", "cuda_gpu_kern_sum"))
        api_rows = tables.get("cuda_api_sum", [])
        counts = _aggregate_nsys(api_rows) if api_rows else {}
        if counts:
            return counts

        kern_rows = tables.get("cuda_gpu_kern_sum", [])
        return _aggregate_nsys(kern_rows)

    raise ValueError(f"Unknown profile_mode: {profile_mode}")
