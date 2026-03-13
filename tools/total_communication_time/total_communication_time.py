"""
Metric: total_communication_time
Description: Total time in communication kernels (NCCL, XLA collectives) in ms.
Unit: ms
Returns: Float >= 0, or -1 if data unavailable

Supported trace types (dispatched via workload YAML):
  nsys        — delegates to total_communication_time_group_9 (nsys stats kernsum)
  json        — reads a single PyTorch-profiler JSON file (rank0_trace.json)
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


def _merge_intervals(intervals: list) -> float:
    """Merge overlapping (start, end) intervals and return total covered length."""
    if not intervals:
        return 0.0
    intervals.sort()
    merged_end = intervals[0][1]
    total = 0.0
    cur_start = intervals[0][0]
    for start, end in intervals[1:]:
        if start <= merged_end:
            merged_end = max(merged_end, end)
        else:
            total += merged_end - cur_start
            cur_start, merged_end = start, end
    total += merged_end - cur_start
    return total


def _calc_json(directory: str) -> float:
    """
    Total communication time from PyTorch-profiler JSON files (kineto_trace_*.json).
    Merges overlapping comm kernel intervals per rank, then averages across ranks.
    Falls back to rank0_trace.json or first .json if no kineto files found.
    Returns seconds.
    """
    _all_json = sorted(fn for fn in os.listdir(directory) if fn.endswith(".json"))
    _kineto = sorted(fn for fn in _all_json if fn.startswith("kineto_trace_"))
    MAX_RANKS = 8
    if _kineto:
        paths = [os.path.join(directory, fn) for fn in _kineto[:MAX_RANKS]]
    else:
        rank0 = os.path.join(directory, "rank0_trace.json")
        if os.path.exists(rank0):
            paths = [rank0]
        elif _all_json:
            paths = [os.path.join(directory, _all_json[0])]
        else:
            print(f"Error: No JSON files found in {directory}", file=sys.stderr)
            return -1

    rank_times = []
    for path in paths:
        comm_intervals = []
        any_data = False

        for e in _load_json_events(path):
            if (
                isinstance(e, dict)
                and e.get("ph") == "X"
                and e.get("cat") == "kernel"
            ):
                ts = e.get("ts")
                dur = e.get("dur")
                if ts is None or dur is None:
                    continue
                any_data = True
                if _COMM_RE.search(e.get("name", "")):
                    ts = float(ts)
                    comm_intervals.append((ts, ts + float(dur)))

        if any_data:
            rank_times.append(_merge_intervals(comm_intervals) / 1e6)  # µs → s

    if not rank_times:
        print(f"Error: No kernel data found in {directory}", file=sys.stderr)
        return -1

    return sum(rank_times) / len(rank_times)


def _calc_tpu(directory: str) -> float:
    import glob
    import importlib.util

    trace_candidates = glob.glob(os.path.join(directory, "*.trace.json"))

    if not trace_candidates:
        trace_candidates = glob.glob(os.path.join(directory, "**/*.json"), recursive=True)

    if not trace_candidates:
        print(f"No TPU trace JSON found in {directory}", file=sys.stderr)
        return -1

    trace_json = trace_candidates[0]

    tools_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    comm_path = os.path.join(tools_dir, "comm_time-group-21", "comm_time.py")

    spec = importlib.util.spec_from_file_location("comm_time_group_21", comm_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        return module.compute_metric(trace_json, "total")
    except Exception as e:
        return -1


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
