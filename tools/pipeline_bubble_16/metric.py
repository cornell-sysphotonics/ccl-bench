from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from tools.common.detect_profile_mode import (
    available_profile_modes,
    list_torch_trace_dirs,
)
from tools.common.hta_wrappers import temporal_breakdown
from tools.common.nsys_stats import run_nsys_stats_in_dir


if TYPE_CHECKING:
    from pathlib import Path


MetricResult = dict[str, Any]


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    if profile_mode == "auto":
        modes = available_profile_modes(directory)
        profile_mode = "torch" if "torch" in modes else modes[0]

    modes = available_profile_modes(directory)

    # Torch traces are required; if absent, error out
    if "torch" not in modes:
        return {"error": "pipeline_bubble_16 requires torch profiler traces"}

    # Torch is present; prefer torch result, but if nsys is also present attach its summary
    if profile_mode not in {"torch", "nsys", "auto"}:
        profile_mode = "torch"

    trace_dirs = list_torch_trace_dirs(directory)
    if not trace_dirs:
        return {"error": "No torch trace JSON files found (expected *trace.json)"}

    trace_dir = str(trace_dirs[0])

    torch_result: dict[str, Any] | None = None
    hta_error: str | None = None

    try:
        tb = temporal_breakdown(trace_dir)
        bubble = {}
        for rank, entry in tb.items():
            idle = float(entry.get("idle_time", 0.0))
            compute = float(entry.get("compute_time", 0.0))
            noncompute = float(entry.get("non_compute_time", entry.get("noncompute_time", 0.0)))
            total = idle + compute + noncompute
            bubble[rank] = idle / total if total > 0 else 0.0

        if bubble:
            avg = sum(bubble.values()) / len(bubble)
            torch_result = {
                "bubble_fraction_per_rank": bubble,
                "avg_bubble_fraction": avg,
                "method": "hta",
            }
    except Exception as e:  # pragma: no cover
        hta_error = str(e)

    if torch_result is None:
        heuristic = _heuristic_bubble(trace_dirs)
        if heuristic:
            if hta_error:
                heuristic["warning"] = f"HTA temporal_breakdown unavailable: {hta_error}"
            torch_result = heuristic

    if torch_result is None:
        return {"error": "No GPU kernel events found and heuristic bubble estimation failed"}

    # If nsys traces are also available, attach their NVTX summary for reference
    if "nsys" in modes:
        nsys_info = _bubble_from_nsys(directory)
        torch_result["nsys_nvtxsum"] = nsys_info

    return torch_result


def _has_gpu_kernels(trace_dirs: list[Path]) -> bool:
    for trace_dir in trace_dirs:
        for tf in trace_dir.glob("*trace.json"):
            data = _load_trace_json(tf)
            if data is None:
                continue
            for ev in data.get("traceEvents", []):
                if ev.get("cat") == "kernel":
                    return True
    return False


def _heuristic_bubble(trace_dirs: list[Path]) -> dict[str, Any] | None:
    per_rank: dict[int, tuple[float, float]] = {}

    for trace_dir in trace_dirs:
        for tf in trace_dir.glob("*trace.json"):
            rank = _infer_rank(tf.name)
            if rank is None:
                continue
            data = _load_trace_json(tf)
            if data is None:
                continue

            events = data.get("traceEvents", []) or []
            intervals = []
            profiler_durs = []

            for ev in events:
                ts = ev.get("ts", 0)
                dur = ev.get("dur", 0)
                if ts is None or dur is None or dur <= 0:
                    continue

                name = ev.get("name", "") or ""
                cat = ev.get("cat", "") or ""
                lower = name.lower()

                if name.startswith("ProfilerStep#"):
                    profiler_durs.append(float(dur))
                if cat in {"cpu_op", "fwdbwd"} or (
                    cat == "user_annotation"
                    and ("nccl" in lower or "c10d::" in lower or "pipeline" in lower)
                ):
                    intervals.append((float(ts), float(ts + dur)))

            if not intervals:
                continue

            intervals.sort(key=lambda x: x[0])
            merged = _merge_intervals(intervals)
            busy = sum(e - s for s, e in merged)

            total = max(profiler_durs) if profiler_durs else merged[-1][1] - merged[0][0]

            if total <= 0:
                continue

            bubble_time = max(0.0, total - busy)
            if rank not in per_rank:
                per_rank[rank] = (bubble_time, total)
            else:
                prev_bubble, prev_total = per_rank[rank]
                per_rank[rank] = (prev_bubble + bubble_time, prev_total + total)

    if not per_rank:
        return None

    bubble_fraction = {r: (b / t if t > 0 else 0.0) for r, (b, t) in per_rank.items()}
    avg = sum(bubble_fraction.values()) / len(bubble_fraction)
    return {
        "bubble_fraction_per_rank": bubble_fraction,
        "avg_bubble_fraction": avg,
        "method": "heuristic",
        "note": "Estimated from CPU/user_annotation gaps; NVTX stage tags recommended for accuracy",
    }


_RANK_RE = re.compile(r"rank(\d+)")


def _infer_rank(name: str) -> int | None:
    m = _RANK_RE.search(name.lower())
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    parts = [p for p in re.split(r"[^0-9]+", name) if p]
    if parts:
        try:
            return int(parts[-1])
        except Exception:
            return None
    return None


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def _bubble_from_nsys(trace_dir: str) -> dict[str, Any]:
    """Approximate bubble using NVTX summaries from Nsight Systems.

    Uses `nsys stats --report nvtxsum` and reports ratio of non-NVTX time to total NVTX time.
    This is an approximation; for accuracy, use torch profiler with CUDA kernels and NVTX stage tags.
    """
    try:
        tables = run_nsys_stats_in_dir(trace_dir, reports=("nvtxsum",))
    except Exception as e:
        return {"error": f"Failed to read nsys nvtxsum report: {e}"}

    rows = tables.get("nvtxsum") or []
    if not rows:
        return {"error": "No nvtxsum data found in nsys stats"}

    dur_fields = ["Total Time (ns)", "Duration (ns)", "Time (ns)", "Duration"]
    total_nvtx = 0.0
    for row in rows:
        for key in dur_fields:
            val = row.get(key)
            if val:
                try:
                    total_nvtx += float(val)
                    break
                except ValueError:
                    continue

    if total_nvtx <= 0:
        return {"error": "nvtxsum report missing duration fields"}

    # Without full timeline we cannot separate idle; return NVTX timing as supplemental data
    return {
        "method": "nsys_nvtxsum",
        "note": "Idle not derivable from nvtxsum; torch profiler with CUDA kernels + NVTX stage tags recommended for accuracy.",
        "total_nvtx_time_ns": total_nvtx,
        "rows": len(rows),
    }


def _load_trace_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
