"""
Metric: collective_data_volume_*
Description: Total data volume (GB) sent per collective type, measured from
             the first rank's trace file over one profiled step.

             Five metrics are exposed:
               collective_vol_allreduce_gb   — AllReduce bytes
               collective_vol_allgather_gb   — AllGather bytes
               collective_vol_reducescatter_gb — ReduceScatter bytes
               collective_vol_alltoall_gb    — AllToAll / SendRecv bytes
               collective_vol_total_gb       — Sum of all four collectives

             For GPU JSON traces, byte counts are derived from NCCL kernel
             arguments: "In msg nelems" × bytes-per-element(dtype).
             For TPU traces, "bytes_accessed" from XLA HLO events is used.

Unit: Gigabytes (GB)
Returns: Float, or -1.0 if the trace type is unsupported or data unavailable.

Supported trace types:
  json        — PyTorch-profiler Chrome JSON (rank*_trace.json or kineto_trace_*.json)
  tpu_profiler / json_tpu — XLA Chrome-trace JSON
"""

import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from json_sampling import select_json_files

# ── dtype byte widths (matches NCCL kernel argument strings) ──────────────────

_DTYPE_BYTES = {
    "Float": 4, "Float32": 4,
    "Double": 8, "Float64": 8,
    "Half": 2, "Float16": 2, "BFloat16": 2, "BF16": 2,
    "Int": 4, "Int32": 4,
    "Int64": 8, "Long": 8,
    "Int16": 2, "Short": 2,
    "Int8": 1, "Byte": 1,
}

# TPU HLO category → canonical collective name
_TPU_HLO_CATEGORY = {
    "all-reduce":      "allreduce",
    "all-gather":      "allgather",
    "reduce-scatter":  "reducescatter",
    "all-to-all":      "alltoall",
}

_COLLECTIVES = ("allreduce", "allgather", "reducescatter", "alltoall")


# ── YAML helpers ──────────────────────────────────────────────────────────────

def _load_yaml(directory: str) -> dict:
    for fn in os.listdir(directory):
        if fn.endswith(".yaml"):
            with open(os.path.join(directory, fn), encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def _get_trace_types(directory: str) -> list:
    return _load_yaml(directory).get("metric_source", {}).get("traces", [])


# ── Event loading with partial-parse fallback ─────────────────────────────────

def _load_events(trace_path: str) -> list:
    with open(trace_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Truncate trailing garbage and retry
        last_close = raw.rfind("}")
        if last_close == -1:
            return []
        try:
            data = json.loads(raw[: last_close + 1])
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]
    if isinstance(data, list):
        return data
    return []


# ── GPU JSON backend ──────────────────────────────────────────────────────────

def _event_bytes(event: dict) -> int:
    args = event.get("args", {})
    nelems = args.get("In msg nelems")
    dtype  = args.get("dtype")
    if nelems is None or dtype is None:
        return 0
    return int(nelems) * _DTYPE_BYTES.get(dtype, 0)


def _collective_type(event: dict) -> str | None:
    """Return canonical collective name for a GPU kernel event, or None."""
    if event.get("ph") != "X" or event.get("cat") != "kernel":
        return None
    name = event.get("name", "")
    coll = event.get("args", {}).get("Collective name", "")

    if name.startswith(("ncclDevKernel_AllReduce", "ncclKernel_AllReduce")) or \
            coll in ("all_reduce", "allreduce"):
        return "allreduce"
    if name.startswith(("ncclDevKernel_AllGather", "ncclKernel_AllGather")) or \
            coll in ("all_gather", "allgather"):
        return "allgather"
    if name.startswith(("ncclDevKernel_ReduceScatter", "ncclKernel_ReduceScatter")) or \
            coll in ("reduce_scatter", "reducescatter"):
        return "reducescatter"
    # AllToAll is surfaced as SendRecv kernels with Collective name = all_to_allv / alltoall
    if (name.startswith(("ncclDevKernel_SendRecv", "ncclKernel_SendRecv")) and
            coll in ("all_to_allv", "all_to_all", "alltoall")):
        return "alltoall"
    return None


def _get_bytes_json(trace_path: str) -> dict:
    totals = dict.fromkeys(_COLLECTIVES, 0)
    for ev in _load_events(trace_path):
        ctype = _collective_type(ev)
        if ctype:
            totals[ctype] += _event_bytes(ev)
    return totals


# ── TPU backend ───────────────────────────────────────────────────────────────

def _get_bytes_tpu(trace_path: str) -> dict:
    totals = dict.fromkeys(_COLLECTIVES, 0)
    for ev in _load_events(trace_path):
        if ev.get("ph") != "X":
            continue
        collective = _TPU_HLO_CATEGORY.get(ev.get("args", {}).get("hlo_category", ""))
        if collective is None:
            continue
        b = ev.get("args", {}).get("bytes_accessed")
        if b is not None:
            totals[collective] += int(b)
    return totals


# ── Rank-0 file discovery ─────────────────────────────────────────────────────

def _find_rank0_trace(directory: str) -> Path | None:
    d = Path(directory)
    for pat in ("rank0_trace.json", "rank*_trace.json", "kineto_trace_*.json"):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    return None


# ── Core dispatcher ───────────────────────────────────────────────────────────

def _collective_bytes(directory: str) -> dict | None:
    """
    Return per-collective byte totals from rank 0 (or first rank found).
    Returns None if the trace type is unsupported or no trace file is found.
    """
    trace_types = _get_trace_types(directory)

    if "tpu_profiler" in trace_types or "json_tpu" in trace_types:
        # TPU: find any .json in the directory (single file per directory)
        candidates = sorted(Path(directory).glob("*.json"))
        if not candidates:
            print(f"Error: no TPU trace JSON found in {directory}", file=sys.stderr)
            return None
        return _get_bytes_tpu(str(candidates[0]))

    if "json" in trace_types:
        trace_path = _find_rank0_trace(directory)
        if trace_path is None:
            print(f"Error: no rank JSON trace found in {directory}", file=sys.stderr)
            return None
        return _get_bytes_json(str(trace_path))

    print(
        f"Error: collective_data_volume requires json or tpu_profiler trace type, "
        f"got {trace_types}",
        file=sys.stderr,
    )
    return None


# ── Public metric_cal functions ───────────────────────────────────────────────

def _bytes_to_gb(b: int) -> float:
    return b / 1e9


def metric_cal_allreduce(directory: str) -> float:
    totals = _collective_bytes(directory)
    return _bytes_to_gb(totals["allreduce"]) if totals is not None else -1.0


def metric_cal_allgather(directory: str) -> float:
    totals = _collective_bytes(directory)
    return _bytes_to_gb(totals["allgather"]) if totals is not None else -1.0


def metric_cal_reducescatter(directory: str) -> float:
    totals = _collective_bytes(directory)
    return _bytes_to_gb(totals["reducescatter"]) if totals is not None else -1.0


def metric_cal_alltoall(directory: str) -> float:
    totals = _collective_bytes(directory)
    return _bytes_to_gb(totals["alltoall"]) if totals is not None else -1.0


def metric_cal_total(directory: str) -> float:
    totals = _collective_bytes(directory)
    if totals is None:
        return -1.0
    return _bytes_to_gb(sum(totals.values()))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python collective_data_volume.py <trace_directory>")
        sys.exit(1)
    d = sys.argv[1]
    t = _collective_bytes(d)
    if t:
        for k, v in t.items():
            print(f"  {k}: {_bytes_to_gb(v):.4f} GB")
        print(f"  total: {_bytes_to_gb(sum(t.values())):.4f} GB")
