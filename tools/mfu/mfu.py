import json
import os
import re
import statistics
import sys
import yaml

# ── Model architecture lookup (model_family → total parameter count) ──────────
# Used in the standard FLOPs estimate: 6 * N * B * S (forward + backward).
# For MoE models, N should reflect the total (not active) parameter count,
# which gives the theoretical peak utilization.
# Add entries here as new model families are onboarded.
_MODEL_ARCH = {
    "deepseek_v3_16b": {"num_params": 16e9},   # DeepSeek-V3 16B test config
    "llama":           {"num_params":  8e9},    # Llama-3.1-8B
}

# ── Hardware peak BF16 TFLOPs lookup (hardware_model) ─────────────────────────
_HARDWARE_PEAK_TFLOPS = {
    "nvidia_a100": 312.0,
    "nvidia_h100": 989.0,
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


# ── XLA / TPU backend (delegates to mfu_group_4) ──────────────────────────────

def _calc_xla(directory: str) -> float:
    from mfu_group_4.mfu import mfu as mfu_group4
    return mfu_group4(directory)


# ── GPU / JSON backend ────────────────────────────────────────────────────────

_STEP_PATTERN = re.compile(r"ProfilerStep#(\d+)$")


def _calc_json(directory: str, yaml_data: dict) -> float:
    """
    Compute MFU from PyTorch-profiler JSON traces (torchtitan format).
    MFU = (2 * B * S * L * H²) / (step_time * world_size * peak_TFLOPS * 1e12)
    """
    workload = yaml_data.get("workload", {})
    data_cfg  = workload.get("data", {})
    hw_cfg    = workload.get("hardware", {}).get("xpu_spec", {})
    model_cfg = workload.get("model", {})

    batch_size   = data_cfg.get("batch_size") or 1
    seq_len      = data_cfg.get("seq_len") or 1
    world_size   = hw_cfg.get("total_count") or 1
    model_family = (model_cfg.get("model_family") or "").lower()
    hw_model     = (hw_cfg.get("model") or "").lower()

    arch = _MODEL_ARCH.get(model_family)
    if arch is None:
        print(f"[mfu/json] Unknown model_family '{model_family}'; add to _MODEL_ARCH.", file=sys.stderr)
        return -1.0

    num_params  = arch["num_params"]

    peak_tflops = _HARDWARE_PEAK_TFLOPS.get(hw_model)
    if peak_tflops is None:
        print(f"[mfu/json] Unknown hardware_model '{hw_model}'; add to _HARDWARE_PEAK_TFLOPS.", file=sys.stderr)
        return -1.0

    # ── Step time: median of ProfilerStep#N durations across all rank files ──
    rank_files = sorted(
        os.path.join(directory, fn)
        for fn in os.listdir(directory)
        if fn.endswith(".json") and (fn.startswith("rank") or fn.startswith("kineto"))
    )
    if not rank_files:
        print(f"[mfu/json] No rank trace files found in {directory}", file=sys.stderr)
        return -1.0

    # Collect per-rank average step times (exclude first and last step)
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

    step_time_us = statistics.mean(per_rank_avg)
    step_time_s  = step_time_us / 1e6

    # ── MFU ──────────────────────────────────────────────────────────────────
    # Theoretical FLOPs for one forward+backward pass (PaLM convention):
    #   6 * N * B * S  (≈ 3x forward, which is 2 * N * B * S for each of fwd+bwd)
    theoretical_flops = 6.0 * num_params * batch_size * seq_len
    peak_flops_total  = peak_tflops * 1e12 * world_size * step_time_s

    if peak_flops_total <= 0:
        return -1.0

    mfu_pct = (theoretical_flops / peak_flops_total) * 100.0
    return round(mfu_pct, 4)


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
        return _calc_xla(directory)

    if "json" in trace_types:
        return _calc_json(directory, yaml_data)

    print(f"[mfu] No supported trace type in {trace_types}", file=sys.stderr)
    return -1.0
