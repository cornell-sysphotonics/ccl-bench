"""
Trace-derived inference latency metrics.

TTFT is the first inference forward/engine execution duration.
TPOT is the average decode forward/engine execution duration after TTFT.

Units: seconds.
Returns -1 when the workload is not inference or the trace lacks inference
forward markers.
"""

import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from json_sampling import select_json_files


_STEP_PATTERN = re.compile(r"ProfilerStep#(\d+)$")
_VLLM_EXECUTE_CONTEXT_PREFIX = "execute_context_"
_SGLANG_RUN_BATCH_PATTERN = re.compile(r"(^|:) run_batch$")
_XLA_STEP_EVENT_NAME = "$core.py:331 step"


def _load_yaml(directory: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("[inference_latency] PyYAML is required to read workload metadata", file=sys.stderr)
        return {}

    for fn in os.listdir(directory):
        if fn.endswith(".yaml"):
            with open(os.path.join(directory, fn)) as f:
                return yaml.safe_load(f) or {}
    return {}


def _get_trace_types(yaml_data: dict) -> list:
    return yaml_data.get("metric_source", {}).get("traces", [])


def _is_inference(yaml_data: dict) -> bool:
    phase = yaml_data.get("workload", {}).get("model", {}).get("phase", "")
    return str(phase).lower() == "inference"


def _output_len(yaml_data: dict) -> int | None:
    val = yaml_data.get("workload", {}).get("data", {}).get("output_len")
    if val in (None, ""):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


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
            bracket = content.find("[", idx)
            if bracket == -1:
                return []
            partial = content[bracket:]
            data = None
            for suffix in ("]}", "]}}}"):
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


def _decode_durations_us(durs_us: list[float]) -> list[float]:
    if len(durs_us) <= 1:
        return []
    # Match avg_step_time's inference treatment by dropping a trailing
    # cooldown-like execution when there are enough decode passes.
    return durs_us[1:-1] if len(durs_us) > 2 else durs_us[1:]


def _summarize_rank_sequences(rank_sequences: list[list[float]], metric: str) -> float:
    vals = []
    for durs_us in rank_sequences:
        if not durs_us:
            continue
        if metric == "ttft":
            vals.append(durs_us[0])
        elif metric == "tpot":
            decode = _decode_durations_us(durs_us)
            if decode:
                vals.append(statistics.mean(decode))
    if not vals:
        return -1.0
    return statistics.mean(vals) / 1e6


def _calc_xla(directory: str, metric: str) -> float:
    rank_files = select_json_files(directory)
    if not rank_files:
        print(f"[inference_latency/xla] No JSON files in {directory}", file=sys.stderr)
        return -1.0

    rank_sequences = []
    for path in rank_files:
        events = _load_json_events(path)
        step_events = sorted(
            [e for e in events
             if isinstance(e, dict)
             and e.get("ph") == "X"
             and e.get("name", "") == _XLA_STEP_EVENT_NAME
             and "dur" in e],
            key=lambda e: e.get("ts", 0),
        )
        if step_events:
            rank_sequences.append([float(e["dur"]) for e in step_events])

    return _summarize_rank_sequences(rank_sequences, metric)


def _calc_json(directory: str, metric: str) -> float:
    rank_files = select_json_files(directory)
    if not rank_files:
        print(f"[inference_latency/json] No JSON files in {directory}", file=sys.stderr)
        return -1.0

    vllm_rank_sequences = []
    sglang_rank_sequences = []
    for path in rank_files:
        events = _load_json_events(path)

        execute_events = sorted(
            [e for e in events
             if isinstance(e, dict)
             and e.get("ph") == "X"
             and e.get("cat") == "user_annotation"
             and e.get("name", "").startswith(_VLLM_EXECUTE_CONTEXT_PREFIX)
             and "dur" in e],
            key=lambda e: e.get("ts", 0),
        )
        if execute_events:
            vllm_rank_sequences.append([float(e["dur"]) for e in execute_events])

        sglang_run_batch_events = sorted(
            [e for e in events
             if isinstance(e, dict)
             and e.get("ph") == "X"
             and e.get("cat") == "python_function"
             and _SGLANG_RUN_BATCH_PATTERN.search(e.get("name", ""))
             and "dur" in e],
            key=lambda e: e.get("ts", 0),
        )
        if sglang_run_batch_events:
            sglang_rank_sequences.append([float(e["dur"]) for e in sglang_run_batch_events])

    if vllm_rank_sequences:
        return _summarize_rank_sequences(vllm_rank_sequences, metric)
    if sglang_rank_sequences:
        return _summarize_rank_sequences(sglang_rank_sequences, metric)

    print(
        f"[inference_latency/json] No vLLM execute_context or SGLang run_batch events found in {directory}",
        file=sys.stderr,
    )
    return -1.0


def metric_cal(directory: str, metric: str) -> float:
    yaml_data = _load_yaml(directory)
    if not _is_inference(yaml_data):
        return -1.0

    if metric == "tpot":
        output_len = _output_len(yaml_data)
        if output_len is not None and output_len <= 1:
            return -1.0

    trace_types = _get_trace_types(yaml_data)
    if "json_tpu" in trace_types:
        return _calc_xla(directory, metric)
    if "json" in trace_types:
        return _calc_json(directory, metric)

    print(f"[inference_latency] No supported trace type in {trace_types}", file=sys.stderr)
    return -1.0


def ttft_metric_cal(directory: str) -> float:
    return metric_cal(directory, "ttft")


def tpot_metric_cal(directory: str) -> float:
    return metric_cal(directory, "tpot")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[2] not in {"ttft", "tpot"}:
        print("Usage: python inference_latency.py <trace_directory> <ttft|tpot>")
        sys.exit(1)
    print(metric_cal(sys.argv[1], sys.argv[2]))
