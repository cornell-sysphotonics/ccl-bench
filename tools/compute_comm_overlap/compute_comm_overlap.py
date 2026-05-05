"""
Metric: compute_comm_overlap
Description: Percentage of total communication time that is overlapped with
             compute kernels. Higher values indicate more effective pipelining
             of communication behind compute.
Unit: percent (0–100)
Returns: Float in [0, 100], or -1 if data unavailable

Supported trace types (dispatched via workload YAML metric_source.traces):
  json     — reads PyTorch-profiler kineto JSON
             (ProfilerStep#N events, or vLLM execute_context_* / SGLang run_batch ranges)
  json_tpu — reads XLA/TPU Chrome-trace JSON ($core.py:331 step events, or
             StepMarker events for XLA FSDP/TP training traces)
  nsys     — reads Nsight Systems SQLite (CUPTI_ACTIVITY_KIND_KERNEL table);
             computes overlap globally over the full trace (no step boundaries
             in NVTX for these traces)
"""

import json
import re
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from json_sampling import select_json_files
from trace_metric_utils import load_yaml, get_trace_types


# ── JSON loader with partial-parse fallback ───────────────────────────────────

def _load_json_events(path: str) -> list:
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


# ── Interval utilities ────────────────────────────────────────────────────────

_COMM_RE = re.compile(
    r"nccl|"
    r"all[_\-]?reduce|all[_\-]?gather|reduce[_\-]?scatter|"
    r"all[_\-]?to[_\-]?all|broadcast|"
    r"cross[_\-]?device|sendrecv|collective",
    re.IGNORECASE,
)

# XLA-specific comm regex: omits 'broadcast' because in XLA HLO, 'broadcast' is
# a local data-layout op (not a network collective).  All actual XLA collectives
# are covered by all-reduce/all-gather/reduce-scatter/all-to-all/collective.
_XLA_COMM_RE = re.compile(
    r"nccl|"
    r"all[_\-]?reduce|all[_\-]?gather|reduce[_\-]?scatter|"
    r"all[_\-]?to[_\-]?all|"
    r"cross[_\-]?device|sendrecv|collective",
    re.IGNORECASE,
)

# XLA/TPU compute op name fragments (matches utils-group-21.py classify_row)
_XLA_COMPUTE_KEYWORDS = frozenset(
    ["dot", "convolution", "gemm", "matmul", "fusion", "compute", "custom-call"]
)


def _is_comm(name: str) -> bool:
    return bool(_COMM_RE.search(name))


def _is_comm_xla(name: str) -> bool:
    return bool(_XLA_COMM_RE.search(name))


def _is_xla_compute(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in _XLA_COMPUTE_KEYWORDS)


def _merge_intervals(pairs: list) -> list:
    """Merge overlapping (start, end) pairs; input must be pre-sorted."""
    if not pairs:
        return []
    merged = [list(pairs[0])]
    for s, e in pairs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _intersection_time(a: list, b: list) -> float:
    """Total length of intersection between two sorted merged interval lists."""
    total = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if start < end:
            total += end - start
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _overlap_for_step(step_start: float, step_end: float, kernel_events: list) -> tuple:
    """
    Returns (overlap_us, total_comm_us) for a single step window.

    Clips each kernel to [step_start, step_end], separates comm vs. compute,
    merges each set, then returns the intersection length and total comm length.
    """
    comm_ivs = []
    compute_ivs = []
    for e in kernel_events:
        ts = e.get("ts")
        dur = e.get("dur")
        if ts is None or dur is None:
            continue
        s = max(float(ts), step_start)
        en = min(float(ts) + float(dur), step_end)
        if s >= en:
            continue
        if _is_comm(e.get("name", "")):
            comm_ivs.append((s, en))
        else:
            compute_ivs.append((s, en))

    comm_merged = _merge_intervals(sorted(comm_ivs))
    total_comm = sum(en - s for s, en in comm_merged)

    if not comm_ivs or not compute_ivs:
        return 0.0, total_comm

    compute_merged = _merge_intervals(sorted(compute_ivs))
    return _intersection_time(comm_merged, compute_merged), total_comm


# ── JSON / GPU backend ────────────────────────────────────────────────────────

_STEP_PATTERN = re.compile(r"ProfilerStep#(\d+)$")
_VLLM_EXECUTE_CONTEXT_PREFIX = "execute_context_"
# Matches "scheduler.py(N): run_batch" (typical) and bare "run_batch" (future-proof).
# \b avoids matching substrings like "set_prefill_run_batch_start_time".
_SGLANG_RUN_BATCH_PATTERN = re.compile(r"\brun_batch\b")
_XLA_STEP_EVENT_NAME = "$core.py:331 step"


def _calc_json(directory: str) -> float:
    """
    Percentage of communication time overlapped with compute, from PyTorch-profiler JSON.

    Uses the same step-detection priority as avg_step_time:
      1. ProfilerStep#N user annotations (training)
      2. execute_context_* user annotations (vLLM inference)
      3. run_batch python_function events (SGLang inference)

    Trims the first and last step to avoid warmup/cooldown skew. Computes
    overlap/total_comm per rank (summed across steps), then averages across ranks.
    Returns percent in [0, 100].
    """
    rank_files = select_json_files(directory)
    if not rank_files:
        print(f"[compute_comm_overlap/json] No JSON files in {directory}", file=sys.stderr)
        return -1.0

    rank_ratios = []
    for path in rank_files:
        events = _load_json_events(path)

        step_events = sorted(
            [e for e in events
             if isinstance(e, dict)
             and e.get("ph") == "X"
             and e.get("cat") == "user_annotation"
             and _STEP_PATTERN.match(e.get("name", ""))],
            key=lambda e: int(_STEP_PATTERN.match(e["name"]).group(1)),
        )
        if not step_events:
            step_events = sorted(
                [e for e in events
                 if isinstance(e, dict)
                 and e.get("ph") == "X"
                 and e.get("cat") == "user_annotation"
                 and e.get("name", "").startswith(_VLLM_EXECUTE_CONTEXT_PREFIX)],
                key=lambda e: e.get("ts", 0),
            )
        if not step_events:
            step_events = sorted(
                [e for e in events
                 if isinstance(e, dict)
                 and e.get("ph") == "X"
                 and e.get("cat") == "python_function"
                 and _SGLANG_RUN_BATCH_PATTERN.search(e.get("name", ""))],
                key=lambda e: e.get("ts", 0),
            )

        if not step_events:
            continue

        inner = step_events[1:-1] if len(step_events) > 2 else step_events

        kernel_events = [
            e for e in events
            if isinstance(e, dict)
            and e.get("ph") == "X"
            and e.get("cat") == "kernel"
        ]

        if not kernel_events:
            # vLLM/SGLang inference traces often capture only CPU-side events
            # (no ProfilerActivity.CUDA).  GPU-level overlap requires an nsys
            # trace; add "nsys" to metric_source.traces in the workload YAML.
            print(
                f"[compute_comm_overlap/json] {len(step_events)} step event(s) found in"
                f" {os.path.basename(path)} but no GPU kernel events — cannot compute"
                f" GPU-level overlap. Add 'nsys' to metric_source.traces for this metric.",
                file=sys.stderr,
            )
            continue

        total_overlap_us = 0.0
        total_comm_us = 0.0
        for step in inner:
            ts = step.get("ts")
            dur = step.get("dur")
            if ts is None or dur is None:
                continue
            overlap, comm = _overlap_for_step(float(ts), float(ts) + float(dur), kernel_events)
            total_overlap_us += overlap
            total_comm_us += comm

        if total_comm_us > 0:
            rank_ratios.append(total_overlap_us / total_comm_us * 100)

    if not rank_ratios:
        print(
            f"[compute_comm_overlap/json] No usable step/kernel data in {directory}",
            file=sys.stderr,
        )
        return -1.0

    return statistics.mean(rank_ratios)


# ── XLA / TPU backend ─────────────────────────────────────────────────────────

def _calc_xla(directory: str) -> float:
    """
    Percentage of communication time overlapped with compute, from an XLA Chrome-trace JSON.

    Step detection priority:
      1. '$core.py:331 step' events (vLLM inference traces, e.g. group-4)
      2. 'StepMarker' events (XLA FSDP/TP training traces, e.g. group-21);
         only the large StepMarkers (>10% of max duration) are used as actual
         training steps — small ones are overhead markers.

    Uses _is_comm_xla instead of _is_comm to avoid classifying the XLA HLO
    'broadcast' op (a local data-layout op) as a network collective.
    Durations are in microseconds despite displayTimeUnit:ns.
    Returns percent in [0, 100].
    """
    json_files = select_json_files(directory)
    if not json_files:
        print(f"[compute_comm_overlap/xla] No JSON files in {directory}", file=sys.stderr)
        return -1.0

    try:
        with open(json_files[0], encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[compute_comm_overlap/xla] Error reading {json_files[0]}: {e}", file=sys.stderr)
        return -1.0

    events = data.get("traceEvents", []) if isinstance(data, dict) else []

    # Primary: '$core.py:331 step' (vLLM inference, group-4 style)
    step_events = sorted(
        [e for e in events
         if isinstance(e, dict)
         and e.get("ph") == "X"
         and e.get("name") == _XLA_STEP_EVENT_NAME],
        key=lambda e: e.get("ts", 0),
    )

    # Fallback: 'StepMarker' (XLA FSDP/TP training, group-21 style)
    if not step_events:
        all_markers = sorted(
            [e for e in events
             if isinstance(e, dict)
             and e.get("ph") == "X"
             and e.get("name") == "StepMarker"
             and e.get("dur") is not None],
            key=lambda e: e.get("ts", 0),
        )
        if all_markers:
            max_dur = max(e["dur"] for e in all_markers)
            step_events = [e for e in all_markers if e["dur"] > max_dur * 0.1]

    if not step_events:
        print(
            f"[compute_comm_overlap/xla] No step events found in {json_files[0]}",
            file=sys.stderr,
        )
        return -1.0

    inner = step_events[1:-1] if len(step_events) > 2 else step_events

    # No cat filter: XLA traces may have cat=None for all ops.
    # Explicitly classify comm vs. compute so unrelated spans (Python frames,
    # annotations) are excluded from both buckets.
    all_dur_events = [
        e for e in events
        if isinstance(e, dict)
        and e.get("ph") == "X"
        and e.get("ts") is not None
        and e.get("dur") is not None
    ]

    total_overlap_us = 0.0
    total_comm_us = 0.0
    for step in inner:
        ts = step.get("ts")
        dur = step.get("dur")
        if ts is None or dur is None:
            continue
        step_start = float(ts)
        step_end = step_start + float(dur)

        comm_ivs = []
        compute_ivs = []
        for e in all_dur_events:
            s = max(float(e["ts"]), step_start)
            en = min(float(e["ts"]) + float(e["dur"]), step_end)
            if s >= en:
                continue
            name = e.get("name", "")
            if _is_comm_xla(name):
                comm_ivs.append((s, en))
            elif _is_xla_compute(name):
                compute_ivs.append((s, en))

        comm_merged = _merge_intervals(sorted(comm_ivs))
        total_comm_us += sum(en - s for s, en in comm_merged)

        if comm_ivs and compute_ivs:
            compute_merged = _merge_intervals(sorted(compute_ivs))
            total_overlap_us += _intersection_time(comm_merged, compute_merged)

    if total_comm_us == 0:
        print(f"[compute_comm_overlap/xla] No usable step data", file=sys.stderr)
        return -1.0

    return total_overlap_us / total_comm_us * 100


# ── Nsight Systems backend ────────────────────────────────────────────────────

def _calc_nsys(directory: str) -> float:
    """
    Percentage of communication time overlapped with compute, from Nsight Systems SQLite.

    Reads all GPU kernel intervals from CUPTI_ACTIVITY_KIND_KERNEL, separates
    them into comm vs. compute, merges each set, then returns overlap/total_comm*100.
    Computed globally over the full trace (nsys traces have no named step NVTX ranges).
    Returns percent in [0, 100].
    """
    import sqlite3
    from nsys_utils import find_sqlite_file

    sqlite_path = find_sqlite_file(directory)
    if sqlite_path is None:
        print(f"[compute_comm_overlap/nsys] No .sqlite file in {directory}", file=sys.stderr)
        return -1.0

    try:
        conn = sqlite3.connect(sqlite_path)
        strings = dict(conn.execute("SELECT id, value FROM StringIds"))
        rows = conn.execute(
            "SELECT start, end, shortName, demangledName, deviceId "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchall()
        conn.close()
    except Exception as e:
        print(f"[compute_comm_overlap/nsys] SQLite error: {e}", file=sys.stderr)
        return -1.0

    # Separate intervals per device — merging across devices would falsely count
    # cross-device "overlap" and collapse simultaneous per-device comm intervals.
    from collections import defaultdict
    device_comm: dict = defaultdict(list)
    device_compute: dict = defaultdict(list)

    for start, end, short_id, dem_id, device_id in rows:
        if start is None or end is None:
            continue
        name = strings.get(short_id) or strings.get(dem_id) or ""
        iv = (float(start), float(end))
        if _is_comm(name):
            device_comm[device_id].append(iv)
        else:
            device_compute[device_id].append(iv)

    if not device_comm:
        print(f"[compute_comm_overlap/nsys] No comm kernels found", file=sys.stderr)
        return -1.0

    total_comm = 0.0
    total_overlap = 0.0
    for dev_id, comm_ivs in device_comm.items():
        comm_merged = _merge_intervals(sorted(comm_ivs))
        total_comm += sum(e - s for s, e in comm_merged)
        compute_ivs = device_compute.get(dev_id, [])
        if not compute_ivs:
            continue
        compute_merged = _merge_intervals(sorted(compute_ivs))
        total_overlap += _intersection_time(comm_merged, compute_merged)

    if total_comm == 0:
        print(f"[compute_comm_overlap/nsys] No comm time found", file=sys.stderr)
        return -1.0

    return total_overlap / total_comm * 100


# ── Unified entry point ────────────────────────────────────────────────────────

def metric_cal(directory: str) -> float:
    """
    Percentage of total communication time overlapped with compute kernels.
    Dispatches to XLA or JSON backend based on workload YAML trace type.
    Returns percent in [0, 100], or -1 if unavailable.
    """
    yaml_data = load_yaml(directory)
    trace_types = get_trace_types(yaml_data)

    if "nsys" in trace_types:
        return _calc_nsys(directory)

    if "json_tpu" in trace_types:
        return _calc_xla(directory)

    if "json" in trace_types:
        return _calc_json(directory)

    print(f"[compute_comm_overlap] No supported trace type in {trace_types}", file=sys.stderr)
    return -1.0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_comm_overlap.py <trace_directory>")
        sys.exit(1)
    result = metric_cal(sys.argv[1])
    print(result)
