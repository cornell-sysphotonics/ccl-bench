import json
import os
import re
import statistics
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from json_sampling import select_json_files

# ── Hardware peak BF16 TFLOPs lookup (hardware_model) ─────────────────────────
_HARDWARE_PEAK_TFLOPS = {
    "nvidia_a100": 312.0,
    "nvidia_h100": 989.0,
    "tpu_v6e": 918.0,
}


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_yaml(directory: str) -> dict:
    for fn in os.listdir(directory):
        if fn.endswith(".yaml"):
            with open(os.path.join(directory, fn)) as f:
                return yaml.safe_load(f) or {}
    return {}


def _get_trace_types(yaml_data: dict) -> list:
    return yaml_data.get("metric_source", {}).get("traces", [])


def _load_json_events(path: str) -> list:
    """Load traceEvents from a PyTorch-profiler JSON file; partial-parse fallback."""
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
    Compute MFU from an XLA/TPU trace.

    The XLA profiler records model_flops summed across all chips in the cluster,
    so the denominator must be peak_per_chip * total_chip_count (not just tp).
    """
    hw_cfg    = yaml_data.get("workload", {}).get("hardware", {}).get("xpu_spec", {})
    hw_model  = (hw_cfg.get("model") or "").lower()
    total_count = hw_cfg.get("total_count")

    peak_per_chip = _HARDWARE_PEAK_TFLOPS.get(hw_model)
    if peak_per_chip is None:
        print(f"[mfu/xla] Unknown hardware_model '{hw_model}'; add to _HARDWARE_PEAK_TFLOPS.", file=sys.stderr)
        return -1.0
    if not total_count:
        print(f"[mfu/xla] Missing xpu_spec.total_count in YAML for {directory}", file=sys.stderr)
        return -1.0

    from trace_utils import find_trace_files, extract_metrics_from_trace

    traces = find_trace_files(directory)
    if not traces:
        print(f"[mfu/xla] No trace files found in {directory}", file=sys.stderr)
        return -1.0

    _, trace_path = traces[0]
    metrics = extract_metrics_from_trace(trace_path)
    active_tflops = metrics.get("active_tflops")
    if active_tflops is None or active_tflops != active_tflops:  # NaN check
        print(f"[mfu/xla] active_tflops unavailable for {directory}", file=sys.stderr)
        return -1.0

    peak_total = peak_per_chip * total_count
    return round((active_tflops / peak_total) * 100.0, 4)


# ── GPU / JSON backend ────────────────────────────────────────────────────────

_STEP_PATTERN = re.compile(r"ProfilerStep#(\d+)$")


def _calc_json(directory: str, yaml_data: dict) -> float:
    """
    Compute MFU from PyTorch-profiler JSON traces (torchtitan format).

    Formula (from "Efficient Large Scale Language Modeling with Mixtures of Experts"):
      FLOP/token = 6(N - N_emb) + 12·L·H·Q·S
      Observed FLOPS = (FLOP/token) × (tokens/second)
      MFU = Observed FLOPS / (world_size × peak_TFLOPS)

    Where:
      N     = total parameter count         (from YAML model_arch.num_params)
      N_emb = embedding parameter count     (from YAML model_arch.num_params_embedding)
      L     = number of transformer layers  (from YAML model_arch.num_layers)
      H     = per-head dimension (head_dim) (from YAML model_arch.head_dim)
      Q     = number of attention heads     (from YAML model_arch.num_heads)
      S     = sequence length               (from YAML workload.data.seq_len)
      tokens/second = (batch_size × seq_len) / step_time
    """
    workload = yaml_data.get("workload", {})
    data_cfg = workload.get("data", {})
    hw_cfg   = workload.get("hardware", {}).get("xpu_spec", {})
    arch     = workload.get("model", {}).get("model_arch", {})

    if arch is None:
        model_family = workload.get("model", {}).get("model_family", "unknown")
        print(f"[mfu/json] No model_arch in YAML for '{model_family}'; add model_arch section.", file=sys.stderr)
        return -1.0

    # ── Architecture params from YAML ─────────────────────────────────────────
    N     = arch.get("num_params")
    N_emb = arch.get("num_params_embedding", 0) or 0
    L     = arch.get("num_layers")
    H     = arch.get("head_dim")
    Q     = arch.get("num_heads")

    if any(v is None for v in (N, L, H, Q)):
        print(f"[mfu/json] model_arch missing required fields (num_params, num_layers, head_dim, num_heads).", file=sys.stderr)
        return -1.0

    # ── Workload params from YAML ─────────────────────────────────────────────
    batch_size = data_cfg.get("batch_size") or 1
    seq_len    = data_cfg.get("seq_len") or 1
    world_size = hw_cfg.get("total_count") or 1
    hw_model   = (hw_cfg.get("model") or "").lower()

    peak_tflops = _HARDWARE_PEAK_TFLOPS.get(hw_model)
    if peak_tflops is None:
        print(f"[mfu/json] Unknown hardware_model '{hw_model}'; add to _HARDWARE_PEAK_TFLOPS.", file=sys.stderr)
        return -1.0

    # ── Step time from ProfilerStep#N events across rank trace files ──────────
    rank_files = [
        f for f in select_json_files(directory)
        if os.path.basename(f).startswith(("rank", "kineto"))
    ]
    if not rank_files:
        print(f"[mfu/json] No rank trace files found in {directory}", file=sys.stderr)
        return -1.0

    per_rank_avg = []
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
            continue
        durs_us = [e["dur"] for e in step_events if "dur" in e]
        inner = durs_us[1:-1] if len(durs_us) > 2 else durs_us
        if inner:
            per_rank_avg.append(statistics.mean(inner))

    if not per_rank_avg:
        print(f"[mfu/json] No ProfilerStep events found in {directory}", file=sys.stderr)
        return -1.0

    step_time_s = statistics.mean(per_rank_avg) / 1e6  # µs → s

    # ── MFU ──────────────────────────────────────────────────────────────────
    flop_per_token  = 6 * (N - N_emb) + 12 * L * H * Q * seq_len
    tokens_per_sec  = (batch_size * seq_len) / step_time_s
    observed_flops  = flop_per_token * tokens_per_sec
    peak_flops_total = peak_tflops * 1e12 * world_size

    if peak_flops_total <= 0:
        return -1.0

    return round((observed_flops / peak_flops_total) * 100.0, 4)


# ── Unified entry point ────────────────────────────────────────────────────────

def metric_cal(directory: str) -> float:
    """
    Calculate Model FLOPs Utilization (MFU).
    Dispatches to XLA or GPU backend based on the workload YAML trace type.
    Returns MFU percentage, or -1.0 if unavailable.
    """
    yaml_data   = _load_yaml(directory)
    trace_types = _get_trace_types(yaml_data)

    if "xla_trace" in trace_types:
        return _calc_xla(directory, yaml_data)

    if "json" in trace_types:
        return _calc_json(directory, yaml_data)

    print(f"[mfu] No supported trace type in {trace_types}", file=sys.stderr)
    return -1.0
