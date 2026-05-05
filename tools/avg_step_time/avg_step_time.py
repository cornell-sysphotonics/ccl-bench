"""
Metric: avg_step_time
Description: Average wall-clock time per training/inference step.
Unit: seconds
Returns: Float >= 0, or -1 if data unavailable

Supported trace types (dispatched via workload YAML metric_source.traces):
  json_tpu — reads TPU profiler Chrome-trace JSON ($core.py:331 step events)
  json      — reads PyTorch-profiler kineto JSON
              (ProfilerStep#N events, or vLLM execute_context_* ranges)
"""

import json
import os
import re
import statistics
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from json_sampling import select_json_files


# ── YAML helpers ───────────────────────────────────────────────────────────────

def _load_yaml(directory: str) -> dict:
    for fn in os.listdir(directory):
        if fn.endswith(".yaml"):
            with open(os.path.join(directory, fn)) as f:
                return yaml.safe_load(f) or {}
    return {}


def _get_trace_types(yaml_data: dict) -> list:
    return yaml_data.get("metric_source", {}).get("traces", [])


# ── JSON loader with partial-parse fallback ────────────────────────────────────

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


# ── XLA / TPU backend ─────────────────────────────────────────────────────────

def _calc_xla(directory: str, yaml_data: dict) -> float:
    """
    Step time from an XLA Chrome-trace JSON.

    Primary path: '$core.py:331 step' events (vLLM inference traces, e.g. group-4).
    Durations are in microseconds (Chrome trace format default), despite displayTimeUnit:ns label.

    Fallback: total trace wall time / iteration count from the YAML.
    XLA client-side traces for training (e.g. FSDP) only record async dispatch
    events — the real step time must be derived from the total profiled window.
    Returns seconds.
    """
    json_files = select_json_files(directory)
    if not json_files:
        print(f"[avg_step_time/xla] No JSON files in {directory}", file=sys.stderr)
        return -1.0

    STEP_EVENT_NAME = "$core.py:331 step"
    try:
        with open(json_files[0], encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[avg_step_time/xla] Error reading {json_files[0]}: {e}", file=sys.stderr)
        return -1.0

    events = data.get("traceEvents", []) if isinstance(data, dict) else []
    step_events = [
        e for e in events
        if e.get("ph") == "X" and e.get("name", "") == STEP_EVENT_NAME
    ]

    if step_events:
        durs_us = [e["dur"] for e in step_events if "dur" in e]
        inner = durs_us[1:-1] if len(durs_us) > 2 else durs_us
        return statistics.mean(inner) / 1e6  # µs → s (XLA trace dur in µs despite displayTimeUnit:ns)

    # Second path: jit_train_step events (TorchXLA training traces).
    # Each step has one event per device pid; use the min pid as representative.
    # Guard: if YAML iteration > jit event count, multi-step fusing is in use
    # (XLA batches multiple training steps per compiled call) — fall through to
    # the wall_time/iteration fallback which uses the correct total count.
    jit_events = sorted(
        [e for e in events
         if e.get("ph") == "X"
         and isinstance(e.get("name"), str)
         and e["name"].startswith("jit_train_step")
         and "dur" in e],
        key=lambda e: e.get("ts", 0),
    )
    if jit_events:
        first_pid = min(e["pid"] for e in jit_events)
        pid_events = [e for e in jit_events if e["pid"] == first_pid]
        num_iter = yaml_data.get("workload", {}).get("model", {}).get("iteration") or 0
        if not num_iter or num_iter <= len(pid_events):
            pid_durs = [e["dur"] for e in pid_events]
            inner = pid_durs[1:-1] if len(pid_durs) > 2 else pid_durs
            if inner:
                return statistics.mean(inner) / 1e6

    # Fallback: wall_time / iteration_count
    # XLA client traces for training record async dispatches only; real step time
    # = total profiled wall time / number of iterations.
    num_iter = yaml_data.get("workload", {}).get("model", {}).get("iteration")
    if not num_iter:
        print(f"[avg_step_time/xla] No step events and no iteration count in YAML", file=sys.stderr)
        return -1.0

    from trace_utils import find_trace_files, extract_metrics_from_trace
    traces = find_trace_files(directory)
    if not traces:
        return -1.0
    metrics = extract_metrics_from_trace(traces[0][1])  # timestamps treated as µs
    wall_s = metrics.get("wall_s", 0.0)
    if not wall_s:
        return -1.0
    return round(wall_s / num_iter, 6)


# ── JSON / GPU backend ────────────────────────────────────────────────────────

_STEP_PATTERN = re.compile(r"ProfilerStep#(\d+)$")
_VLLM_EXECUTE_CONTEXT_PREFIX = "execute_context_"
_SGLANG_RUN_BATCH_PATTERN = re.compile(r"(^|:) run_batch$")


def _calc_json(directory: str) -> float:
    """
    Step time from PyTorch-profiler kineto JSON.

    Training traces normally contain ProfilerStep#N user annotations.
    vLLM inference traces do not; they contain execute_context_* user ranges,
    where each range is one engine execution/decode iteration.  Use those as
    the inference step marker rather than the benchmark's full batch latency.

    Averages across all sampled rank files.
    Returns seconds.
    """
    rank_files = select_json_files(directory)
    if not rank_files:
        print(f"[avg_step_time/json] No JSON files in {directory}", file=sys.stderr)
        return -1.0

    profiler_step_rank_avgs = []
    vllm_execute_rank_avgs = []
    sglang_run_batch_rank_avgs = []
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
        if step_events:
            durs_us = [e["dur"] for e in step_events if "dur" in e]
            inner = durs_us[1:-1] if len(durs_us) > 2 else durs_us
            if inner:
                profiler_step_rank_avgs.append(statistics.mean(inner))

        execute_events = sorted(
            [e for e in events
             if isinstance(e, dict)
             and e.get("ph") == "X"
             and e.get("cat") == "user_annotation"
             and e.get("name", "").startswith(_VLLM_EXECUTE_CONTEXT_PREFIX)],
            key=lambda e: e.get("ts", 0),
        )
        durs_us = [e["dur"] for e in execute_events if "dur" in e]
        inner = durs_us[1:-1] if len(durs_us) > 2 else durs_us
        if inner:
            vllm_execute_rank_avgs.append(statistics.mean(inner))

        sglang_run_batch_events = sorted(
            [e for e in events
             if isinstance(e, dict)
             and e.get("ph") == "X"
             and e.get("cat") == "python_function"
             and _SGLANG_RUN_BATCH_PATTERN.search(e.get("name", ""))],
            key=lambda e: e.get("ts", 0),
        )
        durs_us = [e["dur"] for e in sglang_run_batch_events if "dur" in e]
        inner = durs_us[1:-1] if len(durs_us) > 2 else durs_us
        if inner:
            sglang_run_batch_rank_avgs.append(statistics.mean(inner))

    if profiler_step_rank_avgs:
        return statistics.mean(profiler_step_rank_avgs) / 1e6  # µs → s

    if vllm_execute_rank_avgs:
        return statistics.mean(vllm_execute_rank_avgs) / 1e6  # µs → s

    if sglang_run_batch_rank_avgs:
        return statistics.mean(sglang_run_batch_rank_avgs) / 1e6  # µs → s

    print(
        f"[avg_step_time/json] No ProfilerStep, vLLM execute_context, or SGLang run_batch events found in {directory}",
        file=sys.stderr,
    )
    return -1.0


# ── Unified entry point ────────────────────────────────────────────────────────

def metric_cal(directory: str) -> float:
    """
    Average step time in seconds.
    Dispatches to XLA or JSON backend based on workload YAML trace type.
    Returns seconds, or -1 if unavailable.
    """
    yaml_data   = _load_yaml(directory)
    trace_types = _get_trace_types(yaml_data)

    if "json_tpu" in trace_types:
        return _calc_xla(directory, yaml_data)

    if "json" in trace_types:
        return _calc_json(directory)

    print(f"[avg_step_time] No supported trace type in {trace_types}", file=sys.stderr)
    return -1.0
