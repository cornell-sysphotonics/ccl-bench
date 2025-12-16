from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

from tools.common.detect_profile_mode import detect_profile_mode
from tools.common.hta_wrappers import kernel_breakdown, temporal_breakdown
from tools.common.math_utils import compute_mfu, tokens_per_second_per_gb, tokens_per_step_per_gb
from tools.common.metadata import (
    flops_per_token,
    hbm_bytes,
    load_run_metadata,
    peak_flops_per_gpu,
    tokens_per_step,
    world_size,
)
from tools.throughput_tokens_16.metric import metric_cal as throughput_metric


MetricResult = dict[str, Any]

_LOGGER = logging.getLogger(__name__)


_DEF_MEM_STATS = ("memory_stats.json", "memory_snapshot.json")


def _parse_memory_file(path: Path) -> dict[str, Any] | None:
    try:
        return cast("dict[str, Any]", json.loads(path.read_text()))
    except Exception as exc:
        _LOGGER.warning("Failed to parse memory stats from %s: %s", path, exc)
        return None


def _load_memory_stats(trace_dir: str) -> dict[str, Any]:
    for name in _DEF_MEM_STATS:
        path = Path(trace_dir) / name
        if not path.exists():
            continue
        parsed = _parse_memory_file(path)
        if parsed is not None:
            return parsed
    return {}


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    if profile_mode == "auto":
        profile_mode = detect_profile_mode(directory)

    meta = load_run_metadata(directory)
    tp_step = tokens_per_step(meta)
    world = world_size(meta)

    throughput = throughput_metric(directory, profile_mode=profile_mode)
    tokens_per_sec = float(throughput.get("throughput_tokens_per_sec", 0.0))

    mfu_val = compute_mfu(tokens_per_sec, flops_per_token(meta), world, peak_flops_per_gpu(meta))

    sm_util = None
    kernels = None
    if profile_mode == "torch":
        tb = temporal_breakdown(directory)
        if tb:
            # tb is per-rank dict
            fractions = []
            for entry in tb.values():
                idle = float(entry.get("idle_time", 0.0))
                compute = float(entry.get("compute_time", 0.0))
                noncompute = float(entry.get("non_compute_time", entry.get("noncompute_time", 0.0)))
                total = idle + compute + noncompute
                if total > 0:
                    fractions.append(compute / total)
            if fractions:
                sm_util = sum(fractions) / len(fractions)
        try:
            kernels = kernel_breakdown(directory)
        except Exception:
            kernels = None

    mem_stats = _load_memory_stats(directory)
    per_rank_memory = {}
    hbm = hbm_bytes(meta)
    for rank_str, stats in mem_stats.items():
        if not isinstance(stats, dict):
            continue
        peak_alloc_raw = stats.get("peak_alloc_bytes", stats.get("max_memory_allocated"))
        peak_reserved_raw = stats.get("peak_reserved_bytes", stats.get("max_memory_reserved"))
        peak_alloc = int(peak_alloc_raw) if peak_alloc_raw is not None else 0
        peak_reserved = int(peak_reserved_raw) if peak_reserved_raw is not None else 0
        efficiency = (peak_alloc / peak_reserved) if peak_reserved else 0.0
        per_rank_memory[rank_str] = {
            "peak_alloc_bytes": peak_alloc,
            "peak_reserved_bytes": peak_reserved,
            "memory_efficiency": efficiency,
            "tokens_per_second_per_gb": tokens_per_second_per_gb(
                tokens_per_sec / max(world, 1), peak_alloc or 1
            ),
            "tokens_per_step_per_gb": tokens_per_step_per_gb(
                tp_step / max(world, 1), peak_alloc or 1
            ),
            "hbm_utilization": (peak_alloc / hbm) if hbm else 0.0,
        }

    return {
        "mfu": mfu_val,
        "sm_util_fraction": sm_util,
        "throughput": throughput,
        "per_rank_memory": per_rank_memory,
        "kernel_breakdown": kernels,
    }
