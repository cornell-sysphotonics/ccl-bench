"""
Metric: communication_ratio
Description: Percentage of total kernel time spent in communication kernels
             (NCCL, cross-device, XLA collectives). Similar to
             communication_fraction but uses summed (non-merged) kernel
             durations as the denominator.
Unit: Percentage (%)
Returns: Float between 0-100, or -1 if data unavailable

Supported trace types (dispatched via workload YAML):
  nsys        — delegates to communication_ratio_group_9 (nsys stats kernsum)
  json        — reads PyTorch-profiler JSON files (rank0_trace.json, …)
  tpu_profiler — reads TPU profiler Chrome-trace JSON (XLA collective ops)
"""

import json
import os
import re
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from json_sampling import select_json_files


# ── Communication kernel patterns ────────────────────────────────────────────

_COMM_RE = re.compile(
    r"^nccl|ncclDevKernel|ncclKernel|"
    r"allreduce|allgather|reducescatter|broadcast|"
    r"cross[_\-]?device|all_reduce|all_gather|reduce_scatter|"
    r"sendrecv",
    re.IGNORECASE,
)


def _is_comm(name: str) -> bool:
    return bool(_COMM_RE.search(name))


# ── YAML helper ──────────────────────────────────────────────────────────────

def _load_yaml(directory: str) -> dict:
    for fn in os.listdir(directory):
        if fn.endswith(".yaml"):
            with open(os.path.join(directory, fn)) as f:
                return yaml.safe_load(f) or {}
    return {}


def _get_trace_types(directory: str) -> list:
    return _load_yaml(directory).get("metric_source", {}).get("traces", [])


# ── NSYS backend ─────────────────────────────────────────────────────────────

def _calc_nsys(directory: str) -> float:
    from communication_ratio_group_9.communication_ratio_group_9 import compute_comm_ratio
    return compute_comm_ratio(directory)


# ── JSON backend ──────────────────────────────────────────────────────────────

def _load_json_events(path: str):
    """Load traceEvents from a PyTorch-profiler JSON file; returns list or []."""
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
            bracket = content.find('[', idx)
            if bracket == -1:
                return []
            partial = content[bracket:]
            data = None
            for suffix in (']}', ']}}}'):
                try:
                    data = json.loads(partial + suffix)
                    break
                except json.JSONDecodeError:
                    pass
            if data is None:
                return []
            if isinstance(data, list):
                return data
        except Exception:
            return []

    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return []


def _calc_json(directory: str) -> float:
    """
    Communication ratio from PyTorch-profiler JSON files.
    For each rank: comm_kernel_time / total_kernel_time (summed, non-merged).
    Returns weighted aggregate (sum comm / sum total) across all ranks.
    """
    json_files = select_json_files(directory)
    if not json_files:
        print(f"Error: No JSON files found in {directory}", file=sys.stderr)
        return -1

    total_sum = 0.0
    comm_sum = 0.0
    any_data = False

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
                total_sum += dur
                any_data = True
                if _is_comm(e.get("name", "")):
                    comm_sum += dur

    if not any_data or total_sum == 0:
        print(f"Error: No usable kernel data in {directory}", file=sys.stderr)
        return -1

    return float((comm_sum / total_sum) * 100.0)


# ── TPU profiler backend ──────────────────────────────────────────────────────

_TPU_COMM_OPS = {
    "all-reduce", "allreduce", "all_reduce",
    "all-gather", "allgather", "all_gather",
    "all-to-all", "alltoall",
    "reduce-scatter", "reducescatter", "reduce_scatter",
    "collective-permute", "collective-permute-start", "collective-permute-done",
    "send", "recv", "broadcast",
}


def _calc_tpu(directory: str) -> float:
    """
    Communication ratio from TPU profiler Chrome-trace JSON.
    Aggregated comm op time / total XLA op time across all TPU devices.
    """
    json_files = sorted(
        os.path.join(directory, fn)
        for fn in os.listdir(directory)
        if fn.endswith(".json")
    )
    if not json_files:
        print(f"Error: No JSON files found in {directory}", file=sys.stderr)
        return -1

    try:
        with open(json_files[0], encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_files[0]}: {e}", file=sys.stderr)
        return -1

    events = data.get("traceEvents", []) if isinstance(data, dict) else []

    tpu_pids: set = set()
    for e in events:
        if (
            isinstance(e, dict)
            and e.get("ph") == "M"
            and e.get("name") == "process_name"
            and "/device:TPU:" in e.get("args", {}).get("name", "")
        ):
            tpu_pids.add(e["pid"])

    total_dur = 0.0
    comm_dur = 0.0

    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        if tpu_pids and e.get("pid") not in tpu_pids:
            continue
        args = e.get("args", {}) or {}
        dev_ps = args.get("device_duration_ps")
        dur = float(dev_ps) / 1000.0 if dev_ps is not None else float(e.get("dur") or 0)
        if dur <= 0:
            continue
        total_dur += dur
        name = (e.get("name") or "").lower()
        hlo_cat = (args.get("hlo_category") or "").lower()
        if name in _TPU_COMM_OPS or hlo_cat in _TPU_COMM_OPS:
            comm_dur += dur

    if total_dur == 0:
        print(f"Error: No TPU device events found in {directory}", file=sys.stderr)
        return -1

    return float((comm_dur / total_dur) * 100.0)


# ── Dispatcher ────────────────────────────────────────────────────────────────

def metric_cal(directory: str) -> float:
    """
    Calculate communication ratio.

    Dispatches to the appropriate backend based on the workload YAML's
    metric_source.traces field.

    Args:
        directory: Path to the trace directory (must contain a workload YAML).

    Returns:
        float: Communication ratio percentage (0–100), or -1 if unavailable.
    """
    trace_types = _get_trace_types(directory)

    if "nsys" in trace_types:
        return _calc_nsys(directory)
    elif "json" in trace_types:
        return _calc_json(directory)
    elif "tpu_profiler" in trace_types:
        return _calc_tpu(directory)
    else:
        print(
            f"Error: unsupported trace types {trace_types} for communication_ratio",
            file=sys.stderr,
        )
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python communication_ratio.py <trace_directory>")
        sys.exit(1)
    result = metric_cal(sys.argv[1])
    print(result)
