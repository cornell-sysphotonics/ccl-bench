from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from tools.common.detect_profile_mode import detect_profile_mode, list_torch_trace_dirs
from tools.common.nsys_stats import run_nsys_stats_in_dir


MetricResult = dict[str, Any]


DTYPE_BYTES: dict[str, int] = {
    "float": 4,
    "float32": 4,
    "half": 2,
    "float16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "double": 8,
    "float64": 8,
    "int": 4,
    "int32": 4,
    "int64": 8,
    "long": 8,
    "short": 2,
    "bool": 1,
}


def _dtype_size(dtype: str | None) -> int | None:
    if not dtype:
        return None
    key = dtype.replace("c10::", "").lower()
    return DTYPE_BYTES.get(key)


def _bytes_from_torch_traces(trace_dirs: list[str]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)

    for trace_dir in trace_dirs:
        for tf in Path(trace_dir).glob("*trace.json"):
            trace = _load_trace_json(tf)
            if trace is None:
                continue

            for ev in trace.get("traceEvents", []):
                name = ev.get("name", "")
                if "nccl" not in name.lower():
                    continue

                args = ev.get("args", {}) or {}
                types = args.get("Input type") or args.get("Input Type") or []
                dims_list = args.get("Input Dims") or args.get("Input dims") or []

                # Normalize to list form
                types_list = types if isinstance(types, list) else [types]
                dims_outer = dims_list if isinstance(dims_list, list) else []

                for idx, dims in enumerate(dims_outer):
                    dtype = (
                        types_list[idx]
                        if idx < len(types_list)
                        else (types_list[0] if types_list else None)
                    )
                    elem_size = _dtype_size(dtype if isinstance(dtype, str) else None)
                    if elem_size is None or not isinstance(dims, list):
                        continue

                    num_elems = 1
                    ok = True
                    for d in dims:
                        try:
                            num = int(d)
                        except Exception:
                            ok = False
                            break
                        if num <= 0:
                            ok = False
                            break
                        num_elems *= num

                    if not ok:
                        continue

                    bytes_val = num_elems * elem_size
                    totals[name] += float(bytes_val)

    totals["total_bytes"] = sum(v for k, v in totals.items() if k != "total_bytes")
    return dict(totals)


def _load_trace_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _bytes_from_nsys(rows: list[dict[str, str]]) -> dict[str, float]:
    """Extract byte sizes from nsys stats rows, when available."""
    totals: dict[str, float] = defaultdict(float)
    size_fields = [
        "Average",
        "Total",
        "Size",
        "Size (B)",
        "Bytes",
        "Total Bytes",
        "Total Data Size (Bytes)",
    ]

    for row in rows:
        name = row.get("Name") or row.get("API Name") or ""
        if "nccl" not in name.lower():
            continue

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
    return dict(totals)


def metric_cal(directory: str, profile_mode: str = "auto") -> MetricResult:
    if profile_mode == "auto":
        profile_mode = detect_profile_mode(directory)

    if profile_mode == "torch":
        trace_dirs = [str(p) for p in list_torch_trace_dirs(directory)]
        if not trace_dirs:
            return {"error": "No torch trace JSON files found (expected *trace.json)"}

        totals = _bytes_from_torch_traces(trace_dirs)
        if totals.get("total_bytes", 0) == 0:
            return {
                **totals,
                "warning": "No NCCL events with Input Dims found in torch traces",
            }
        return totals

    if profile_mode == "nsys":
        # Prefer summed CUDA API report; fall back to raw if present
        tables = run_nsys_stats_in_dir(directory, reports=("cuda_api_sum", "cuda_api"))
        rows = tables.get("cuda_api_sum") or tables.get("cuda_api") or []

        if not rows:
            return {
                "error": "No cuda_api* reports found in nsys stats; cannot compute comm volume",
            }

        totals = _bytes_from_nsys(rows)
        if totals.get("total_bytes", 0) == 0:
            return {
                "error": "Nsight Systems stats do not include byte-size columns for NCCL; comm volume unavailable",
                "rows_processed": len(rows),
            }

        return totals

    raise ValueError(f"Unknown profile_mode: {profile_mode}")
