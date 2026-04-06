"""
Shared utilities for group_9 memory metrics.

Provides YAML helpers, SQLite file discovery, and Kineto JSON kernel
breakdown used by memory_bound_fraction, memory_transfer_overhead, and
average_memory_bandwidth.
"""

import json
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from json_sampling import select_json_files
from nsys_utils import find_sqlite_file  # noqa: F401 — re-exported for metric modules


# ── YAML helpers ─────────────────────────────────────────────────────────────

def load_yaml(directory: str) -> dict:
    """Find and parse the workload card YAML in *directory*."""
    for fn in os.listdir(directory):
        if fn.endswith(".yaml"):
            with open(os.path.join(directory, fn)) as f:
                return yaml.safe_load(f) or {}
    return {}


def get_trace_types(yaml_data: dict) -> list:
    """Return the ``metric_source.traces`` list from parsed YAML."""
    return yaml_data.get("metric_source", {}).get("traces", [])


# ── Kineto JSON kernel breakdown ─────────────────────────────────────────────

# Patterns used to bucket GPU kernels into categories.
_COMM_RE = re.compile(
    r"nccl|ncclKernel|ncclDevKernel|"
    r"allreduce|allgather|reducescatter|broadcast|"
    r"cross[_\-]?device|all_reduce|all_gather|reduce_scatter|"
    r"sendrecv",
    re.IGNORECASE,
)
_GEMM_RE = re.compile(r"gemm|cutlass|cublas|matmul|triton_mm", re.IGNORECASE)
_SOFTMAX_RE = re.compile(r"softmax", re.IGNORECASE)
_NORM_RE = re.compile(r"layer_?norm|rms_?norm|batch_?norm|group_?norm|norm_kernel", re.IGNORECASE)
_ELEMWISE_RE = re.compile(
    r"elementwise|pointwise|vectorized|unary|binary|aten::(add|mul|div|sub|relu|gelu|silu|sigmoid|tanh|exp|neg|abs|clamp)",
    re.IGNORECASE,
)
_REDUCTION_RE = re.compile(r"reduce(?!scatter)|sum_kernel|mean_kernel", re.IGNORECASE)
_MEMCPY_RE = re.compile(r"memcpy|memset|copy_kernel|Memcpy|Memset", re.IGNORECASE)


def _categorize_kernel(name: str) -> str:
    """Assign a kernel name to one of the breakdown categories."""
    if _COMM_RE.search(name):
        return "communication"
    if _MEMCPY_RE.search(name):
        return "memory_transfer"
    if _GEMM_RE.search(name):
        return "gemm"
    if _SOFTMAX_RE.search(name):
        return "softmax"
    if _NORM_RE.search(name):
        return "normalization"
    if _ELEMWISE_RE.search(name):
        return "elementwise"
    if _REDUCTION_RE.search(name):
        return "reduction"
    return "other"


def _load_json_events(path: str) -> list:
    """Load traceEvents from a PyTorch-profiler JSON file."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            idx = content.find('"traceEvents"')
            if idx == -1:
                return []
            bracket = content.find("[", idx)
            if bracket == -1:
                return []
            partial = content[bracket:]
            for suffix in ("]}",  "]}}}"):
                try:
                    data = json.loads(partial + suffix)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                return []
            if isinstance(data, list):
                return data
        except Exception:
            return []
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return []


def summarize_kineto_kernel_breakdown(directory: str):
    """
    Parse Kineto JSON traces and return a kernel-time breakdown.

    Returns:
        ``(total_kernel_dur, breakdown_dict)`` where *breakdown_dict* maps
        category names (``"gemm"``, ``"communication"``, ``"elementwise"``,
        ``"normalization"``, ``"softmax"``, ``"reduction"``,
        ``"memory_transfer"``, ``"other"``) to summed durations (µs).

        Returns ``None`` if no usable kernel events are found.
    """
    json_files = select_json_files(directory)
    if not json_files:
        print(f"[trace_metric_utils] No JSON files in {directory}", file=sys.stderr)
        return None

    breakdown: dict[str, float] = {}
    total_dur = 0.0

    for path in json_files:
        events = _load_json_events(path)
        for e in events:
            if (
                isinstance(e, dict)
                and e.get("ph") == "X"
                and e.get("cat") == "kernel"
            ):
                dur = e.get("dur")
                if dur is None:
                    continue
                dur = float(dur)
                total_dur += dur
                cat = _categorize_kernel(e.get("name", ""))
                breakdown[cat] = breakdown.get(cat, 0.0) + dur

    if total_dur <= 0:
        return None

    return total_dur, breakdown
