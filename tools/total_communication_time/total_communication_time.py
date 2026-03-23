"""
Metric: total_communication_time
Description: Average communication time per step (NCCL, XLA collectives).
Unit: seconds
Returns: Float >= 0, or -1 if data unavailable

Supported trace types (dispatched via workload YAML):
  nsys        — delegates to total_communication_time_group_9 (nsys stats kernsum)
  json        — reads PyTorch-profiler JSON files (ProfilerStep boundaries)
  tpu_profiler — reads TPU profiler Chrome-trace JSON (XLA collective ops)
"""

import json
import os
import re
import statistics
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from json_sampling import select_json_files


_STEP_PATTERN = re.compile(r"ProfilerStep#(\d+)$")


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
    from nsys_utils import collect_nsys_traces, run_nsys_kernsum_csv, extract_csv_block, parse_kernsum_csv

    _NSYS_COMM_RE = [
        re.compile(r"^nccl", re.IGNORECASE),
        re.compile(r"sendrecv", re.IGNORECASE),
        re.compile(r"cross[_\-]?device", re.IGNORECASE),
        re.compile(r"allgather|reducescatter|allreduce|broadcast", re.IGNORECASE),
    ]

    def _is_nsys_comm(name):
        return any(r.search(name) for r in _NSYS_COMM_RE)

    traces = collect_nsys_traces(directory)
    pth = traces[0]
    out = run_nsys_kernsum_csv(pth)
    rows = parse_kernsum_csv(extract_csv_block(out))
    comm_ns = sum(r["total_ns"] for r in rows if _is_nsys_comm(r["name"]))
    return comm_ns / 1e9  # ns → s


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
    Average communication time per step from PyTorch-profiler JSON files.
    Detects step boundaries via ProfilerStep#N events, computes comm time
    per step (merging overlapping intervals), then averages across inner
    steps and ranks.  Returns seconds.
    """
    json_files = select_json_files(directory)
    if not json_files:
        print(f"Error: No JSON files found in {directory}", file=sys.stderr)
        return -1

    per_rank_avg = []
    for path in json_files:
        events = _load_json_events(path)

        # Collect step boundaries
        step_events = sorted(
            [e for e in events
             if isinstance(e, dict)
             and e.get("ph") == "X"
             and e.get("cat") == "user_annotation"
             and _STEP_PATTERN.match(e.get("name", ""))],
            key=lambda e: int(_STEP_PATTERN.match(e["name"]).group(1)),
        )

        # Collect comm kernel intervals
        comm_intervals = []
        any_data = False
        for e in events:
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
                    ts_f = float(ts)
                    comm_intervals.append((ts_f, ts_f + float(dur)))

        if not any_data:
            continue

        total_comm = _merge_intervals(comm_intervals) / 1e6  # µs → s

        if step_events:
            # Trim first and last steps (warmup/cooldown), like avg_step_time
            inner = step_events[1:-1] if len(step_events) > 2 else step_events
            n_steps = len(inner)

            # Compute comm time only within inner step boundaries
            if len(step_events) > 2:
                inner_start = float(inner[0]["ts"])
                last = inner[-1]
                inner_end = float(last["ts"]) + float(last.get("dur", 0))
                inner_comm = []
                for start, end in comm_intervals:
                    # Clip to inner step window
                    cs = max(start, inner_start)
                    ce = min(end, inner_end)
                    if cs < ce:
                        inner_comm.append((cs, ce))
                total_comm = _merge_intervals(inner_comm) / 1e6

            per_rank_avg.append(total_comm / n_steps)
        else:
            # No step events; return total comm time (single iteration assumed)
            per_rank_avg.append(total_comm)

    if not per_rank_avg:
        print(f"Error: No kernel data found in {directory}", file=sys.stderr)
        return -1

    return statistics.mean(per_rank_avg)


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
        total_s = module.compute_metric(trace_json, "total")
    except Exception:
        return -1

    if total_s is None or total_s < 0:
        return -1

    # Divide by iteration count from YAML to get per-step average
    yaml_data = _load_yaml(directory)
    n_iter = yaml_data.get("workload", {}).get("model", {}).get("iteration", 1)
    if n_iter and n_iter > 0:
        return total_s / n_iter
    return total_s


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
