"""
Metric: total_communication_time
Description: Total time in communication kernels (NCCL, XLA collectives) in ms.
Unit: ms
Returns: Float >= 0, or -1 if data unavailable

Supported trace types (dispatched via workload YAML):
  nsys        — delegates to total_communication_time_group_9 (nsys stats kernsum)
  json        — reads PyTorch-profiler JSON files (rank0_trace.json, …)
  tpu_profiler — reads TPU profiler Chrome-trace JSON (XLA collective ops)
"""

import json
import os
import re
import sys
import yaml


_COMM_RE = re.compile(
    r"^nccl|ncclDevKernel|ncclKernel|"
    r"allreduce|allgather|reducescatter|broadcast|"
    r"cross[_\-]?device|all_reduce|all_gather|reduce_scatter|"
    r"sendrecv",
    re.IGNORECASE,
)

_TPU_COMM_OPS = {
    "all-reduce", "allreduce", "all_reduce",
    "all-gather", "allgather", "all_gather",
    "all-to-all", "alltoall",
    "reduce-scatter", "reducescatter", "reduce_scatter",
    "collective-permute", "collective-permute-start", "collective-permute-done",
    "send", "recv", "broadcast",
}


def _load_yaml(directory: str) -> dict:
    for fn in os.listdir(directory):
        if fn.endswith(".yaml"):
            with open(os.path.join(directory, fn)) as f:
                return yaml.safe_load(f) or {}
    return {}


def _get_trace_types(directory: str) -> list:
    return _load_yaml(directory).get("metric_source", {}).get("traces", [])


def _calc_nsys(directory: str) -> float:
    from total_communication_time_group_9.total_communication_time_group_9 import compute_total_comm_time
    ms = compute_total_comm_time(directory)
    return ms / 1000.0 if ms is not None and ms >= 0 else ms   # ms → s


def _load_json_events(path: str):
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
            data = None
            for suffix in (']}', ']}}}'):
                try:
                    data = json.loads(content[bracket:] + suffix)
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
    Total communication time from PyTorch-profiler JSON files.
    Sums NCCL kernel durations across all rank files, returns ms.
    """
    json_files = sorted(
        os.path.join(directory, fn)
        for fn in os.listdir(directory)
        if fn.endswith(".json")
    )
    if not json_files:
        print(f"Error: No JSON files found in {directory}", file=sys.stderr)
        return -1

    comm_us = 0.0
    any_data = False

    for path in json_files:
        for e in _load_json_events(path):
            if (
                isinstance(e, dict)
                and e.get("ph") == "X"
                and e.get("cat") == "kernel"
            ):
                dur = e.get("dur")
                if dur is None:
                    continue
                any_data = True
                if _COMM_RE.search(e.get("name", "")):
                    comm_us += float(dur)

    if not any_data:
        print(f"Error: No kernel data found in {directory}", file=sys.stderr)
        return -1

    return float(comm_us / 1e6)   # µs → s


def _calc_tpu(directory: str) -> float:
    """
    Total communication time from TPU profiler Chrome-trace JSON (ms).
    Sums XLA collective op durations across all TPU device streams.
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

    comm_ns = 0.0
    any_data = False

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
        any_data = True
        name = (e.get("name") or "").lower()
        hlo_cat = (args.get("hlo_category") or "").lower()
        if name in _TPU_COMM_OPS or hlo_cat in _TPU_COMM_OPS:
            comm_ns += dur

    if not any_data:
        print(f"Error: No TPU device events found in {directory}", file=sys.stderr)
        return -1

    return float(comm_ns / 1e9)   # ns → s


def metric_cal(directory: str) -> float:
    trace_types = _get_trace_types(directory)
    if "nsys" in trace_types:
        return _calc_nsys(directory)
    elif "json" in trace_types:
        return _calc_json(directory)
    elif "tpu_profiler" in trace_types:
        return _calc_tpu(directory)
    else:
        print(
            f"Error: unsupported trace types {trace_types} for total_communication_time",
            file=sys.stderr,
        )
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python total_communication_time.py <trace_directory>")
        sys.exit(1)
    print(metric_cal(sys.argv[1]))
