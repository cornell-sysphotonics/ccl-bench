from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from tools.common.detect_profile_mode import available_profile_modes, list_torch_trace_dirs
from tools.common.hta_wrappers import comm_comp_overlap


if TYPE_CHECKING:
    from pathlib import Path


MetricResult = dict[str, Any]

_COMM_KERNEL_PATTERNS = (
    "nccl",
    "c10d::",
)

_COMPUTE_KERNEL_PATTERNS = (
    "matmul",
    "gemm",
    "cutlass",
    "sgemm",
    "dgemm",
    "hgemm",
    "mma",
    "attention",
    "softmax",
    "layernorm",
    "linear",
    "conv",
    "relu",
    "gelu",
    "aten::",
    "ampere",
    "volta",
    "sm80",
    "sm90",
)

_IGNORE_PATTERNS = (
    "memcpy",
    "memset",
    "cudalaunch",
    "cudadevicesynchronize",
)


def metric_cal(
    directory: str,
    profile_mode: str = "auto",
    *,
    prefer_fallback: bool | None = None,
) -> MetricResult:
    """Compute comm/comp overlap using HTA if available, else trace parsing.

    Set ``prefer_fallback=True`` or use ``profile_mode='torch_fallback'`` to skip HTA and
    force the trace-parsing path (useful for comparison when HTA returns zeros).
    """
    if profile_mode == "auto":
        modes = available_profile_modes(directory)
        profile_mode = "torch" if "torch" in modes else modes[0]

    if profile_mode == "nsys":
        return {"error": "comm_comp_overlap_16 requires torch profiler traces (nsys not supported)"}

    if profile_mode not in {"torch", "torch_fallback", "torch_no_hta", "torch_kernel"}:
        return {"error": "comm_comp_overlap_16 is only supported for torch traces"}

    if prefer_fallback is None:
        prefer_fallback = profile_mode in {"torch_fallback", "torch_no_hta", "torch_kernel"}

    trace_dirs = list_torch_trace_dirs(directory)
    if not trace_dirs:
        return {"error": "No torch trace JSON files found (expected *trace.json)"}

    hta_error: str | None = None

    if not prefer_fallback:
        torch_dir = str(trace_dirs[0])

        # If no GPU kernels exist, HTA will report zeros; skip to fallback in that case
        if _has_gpu_kernels(trace_dirs):
            try:
                hta_result = comm_comp_overlap(torch_dir) | {"method": "hta"}
                # Detect zeroed output (common when HTA cannot find GPU kernels) and fall back
                overlap_values = [
                    v.get("comp_comm_overlap_pctg", 0)
                    for v in hta_result.values()
                    if isinstance(v, dict)
                ]
                if overlap_values and any(val > 0 for val in overlap_values):
                    return hta_result
                hta_error = "HTA returned zero overlap (no GPU kernels in trace)"
            except Exception as hta_err:  # pragma: no cover
                hta_error = str(hta_err)
        else:
            hta_error = "No GPU kernel events found; HTA cannot compute overlap"

    result = _overlap_from_traces(trace_dirs)
    if result is None:
        return {
            "error": "No GPU/kernel events found to compute comm/comp overlap",
            "hta_error": hta_error,
        }
    if hta_error:
        result["warning"] = f"HTA comm_comp_overlap unavailable: {hta_error}"
    return result


def _overlap_from_traces(trace_dirs: list[Path]) -> dict[str, Any] | None:
    total_comm = total_comp = total_overlap = 0.0
    comm_count = comp_count = 0
    method = "kernel"

    for trace_dir in trace_dirs:
        for tf in trace_dir.glob("*trace.json"):
            parsed = _parse_trace(tf)
            if parsed is None:
                continue
            comm_intervals, comp_intervals, counts, used_heuristic = parsed
            if used_heuristic:
                method = "heuristic"

            if not comm_intervals and not comp_intervals:
                continue

            overlap = _calculate_interval_overlap(comm_intervals, comp_intervals)
            comm_dur = sum(e - s for s, e in comm_intervals)
            comp_dur = sum(e - s for s, e in comp_intervals)

            total_comm += comm_dur
            total_comp += comp_dur
            total_overlap += overlap
            comm_count += counts[0]
            comp_count += counts[1]

    if total_comm == 0 and total_comp == 0:
        return None

    comm_time_ms = total_comm / 1000.0
    comp_time_ms = total_comp / 1000.0
    overlap_time_ms = total_overlap / 1000.0
    overlap_ratio_of_comm = total_overlap / total_comm if total_comm > 0 else 0.0
    overlap_ratio_of_comp = total_overlap / total_comp if total_comp > 0 else 0.0

    return {
        "comm_time_ms": comm_time_ms,
        "comp_time_ms": comp_time_ms,
        "overlap_time_ms": overlap_time_ms,
        "overlap_ratio_of_comm": min(overlap_ratio_of_comm, 1.0),
        "overlap_ratio_of_comp": min(overlap_ratio_of_comp, 1.0),
        "num_comm_kernels": comm_count,
        "num_comp_kernels": comp_count,
        "method": method,
    }


def _has_gpu_kernels(trace_dirs: list[Path]) -> bool:
    for trace_dir in trace_dirs:
        for tf in trace_dir.glob("*trace.json"):
            data = _load_trace_json(tf)
            if data is None:
                continue
            if any(ev.get("cat") == "kernel" for ev in data.get("traceEvents", [])):
                return True
    return False


def _parse_trace(
    trace_path: Path,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], tuple[int, int], bool] | None:
    data = _load_trace_json(trace_path)
    if data is None:
        return None

    events = data.get("traceEvents", []) or []
    comm_intervals: list[tuple[float, float]] = []
    comp_intervals: list[tuple[float, float]] = []
    comm_count = comp_count = 0
    used_heuristic = False

    # Prefer GPU kernel events
    for ev in events:
        ts = ev.get("ts", 0)
        dur = ev.get("dur", 0)
        if ts is None or dur is None or dur <= 0:
            continue
        name = ev.get("name", "") or ""
        cat = ev.get("cat", "") or ""

        if cat == "kernel":
            if _is_comm_kernel(name):
                comm_intervals.append((float(ts), float(ts + dur)))
                comm_count += 1
            elif _is_compute_kernel(name):
                comp_intervals.append((float(ts), float(ts + dur)))
                comp_count += 1

    if comm_intervals or comp_intervals:
        return comm_intervals, comp_intervals, (comm_count, comp_count), used_heuristic

    # Heuristic fallback: use cpu_op/fwdbwd as compute, user_annotation with nccl/c10d as comm
    used_heuristic = True
    for ev in events:
        ts = ev.get("ts", 0)
        dur = ev.get("dur", 0)
        if ts is None or dur is None or dur <= 0:
            continue
        name = ev.get("name", "") or ""
        cat = ev.get("cat", "") or ""

        if cat == "user_annotation" and _is_comm_kernel(name):
            comm_intervals.append((float(ts), float(ts + dur)))
            comm_count += 1
        elif cat in {"cpu_op", "fwdbwd"}:
            comp_intervals.append((float(ts), float(ts + dur)))
            comp_count += 1

    if not comm_intervals and not comp_intervals:
        return None

    return comm_intervals, comp_intervals, (comm_count, comp_count), used_heuristic


def _is_comm_kernel(name: str) -> bool:
    lower = name.lower()
    return any(p in lower for p in _COMM_KERNEL_PATTERNS)


def _is_compute_kernel(name: str) -> bool:
    lower = name.lower()
    if any(p in lower for p in _IGNORE_PATTERNS):
        return False
    return any(p in lower for p in _COMPUTE_KERNEL_PATTERNS)


def _calculate_interval_overlap(
    intervals_a: list[tuple[float, float]], intervals_b: list[tuple[float, float]]
) -> float:
    if not intervals_a or not intervals_b:
        return 0.0

    events = []
    for start, end in intervals_a:
        events.append((start, 1, "a"))
        events.append((end, -1, "a"))
    for start, end in intervals_b:
        events.append((start, 1, "b"))
        events.append((end, -1, "b"))

    events.sort(key=lambda x: (x[0], -x[1]))

    overlap = 0.0
    active_a = active_b = 0
    last = events[0][0]
    for time, typ, set_id in events:
        if active_a > 0 and active_b > 0:
            overlap += time - last
        if set_id == "a":
            active_a += typ
        else:
            active_b += typ
        last = time

    return overlap


def _load_trace_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
