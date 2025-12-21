"""
Compute throughput estimation metrics
"""
import sys
import os
import importlib.util
import numpy as np
import pandas as pd
import math
from typing import Dict, Optional

# Import utils-group-21
tools_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_path = os.path.join(tools_dir, "utils-group-21.py")
spec = importlib.util.spec_from_file_location("utils_group_21", utils_path)
utils_group_21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_group_21)
prepare_dataframe = utils_group_21.prepare_dataframe
safe_float = utils_group_21.safe_float
sum_model_flops_from_args = utils_group_21.sum_model_flops_from_args

wall_time_path = os.path.join(tools_dir, "wall_time-group-21", "wall_time.py")
wall_time_spec = importlib.util.spec_from_file_location("wall_time", wall_time_path)
wall_time_module = importlib.util.module_from_spec(wall_time_spec)
wall_time_spec.loader.exec_module(wall_time_module)
compute_wall_time_s = wall_time_module.compute_wall_time_s


def choose_flops_per_token(
    model_params: Optional[float],
    flops_per_token: Optional[float],
    mode: str,
) -> tuple[float, str]:
    note_parts = []

    if flops_per_token is None:
        if model_params is None:
            model_params = 7e9
            note_parts.append("No model_params provided; using default 7e9 (7B parameters).")
        if mode.lower().startswith("train"):
            fpt = 6.0 * float(model_params)
            note_parts.append("Used training heuristic flops/token ≈ 6 * params (dense Transformer).")
        else:
            fpt = 2.0 * float(model_params)
            note_parts.append("Used inference heuristic flops/token ≈ 2 * params (dense Transformer).")
    else:
        fpt = float(flops_per_token)
        note_parts.append("Used user-provided flops_per_token.")

    return float(fpt), " ".join(note_parts)


def estimate_tokens_from_model_flops(
    df: pd.DataFrame,
    wall_time_s: float,
    model_params: float | None = None,
    flops_per_token: float | None = None,
) -> Dict[str, float | str]:
    total_model_flops = sum_model_flops_from_args(df)
    note_parts = []

    fpt, note = choose_flops_per_token(model_params, flops_per_token, "train")
    if note:
        note_parts.append(note)

    if total_model_flops <= 0 or not math.isfinite(total_model_flops):
        ops = df.dropna(subset=["model_flops", "T_s"]).copy()
        ops = ops[(ops["model_flops"] > 0) & (ops["T_s"] > 0)]
        if len(ops) and ops["T_s"].sum() > 0:
            achieved_flops = ops["model_flops"].sum() / ops["T_s"].sum()
            total_model_flops = achieved_flops * wall_time_s if wall_time_s > 0 else 0.0
            note_parts.append("Using achieved FLOPs from trace as fallback since model_flops not in args.")
        else:
            total_model_flops = 0.0
            note_parts.append("No model_flops found in trace args or event data. total_model_flops set to 0.")

    if fpt <= 0 or not math.isfinite(fpt):
        if model_params is not None:
            fpt = 6.0 * float(model_params)
            note_parts.append("Using default training flops/token heuristic (6 * params).")
        else:
            fpt = 0.0
            note_parts.append("Cannot calculate flops_per_token without model_params.")

    estimated_total_tokens = total_model_flops / fpt if fpt > 0 and total_model_flops > 0 else 0.0

    if wall_time_s <= 0 or not math.isfinite(wall_time_s):
        tps = 0.0
        note_parts.append("Invalid wall_time_s; tokens/s set to 0.")
    else:
        tps = estimated_total_tokens / wall_time_s if wall_time_s > 0 else 0.0

    note_parts.append("Estimate depends on model_flops semantics + assumes dense Transformer scaling.")
    return {
        "total_model_flops": float(total_model_flops),
        "flops_per_token_used": float(fpt),
        "estimated_total_tokens": float(estimated_total_tokens),
        "estimated_throughput_tokens_per_s": float(tps) if math.isfinite(tps) else 0.0,
        "estimate_note": " ".join([p for p in note_parts if p]),
    }


def compute_flops_per_token_used(trace_json_path: str, model_params: float | None = None) -> float:
    df = prepare_dataframe(trace_json_path)
    wall_time_s = compute_wall_time_s(df)
    est = estimate_tokens_from_model_flops(df, wall_time_s, model_params=model_params)
    return est["flops_per_token_used"]


def compute_estimated_total_tokens(trace_json_path: str, model_params: float | None = None) -> float:
    df = prepare_dataframe(trace_json_path)
    wall_time_s = compute_wall_time_s(df)
    est = estimate_tokens_from_model_flops(df, wall_time_s, model_params=model_params)
    return est["estimated_total_tokens"]


def compute_estimated_throughput_tokens_per_s(trace_json_path: str, model_params: float | None = None) -> float:
    df = prepare_dataframe(trace_json_path)
    wall_time_s = compute_wall_time_s(df)
    est = estimate_tokens_from_model_flops(df, wall_time_s, model_params=model_params)
    return est["estimated_throughput_tokens_per_s"]


def compute_throughput_estimate_note(trace_json_path: str, model_params: float | None = None) -> str:
    df = prepare_dataframe(trace_json_path)
    wall_time_s = compute_wall_time_s(df)
    est = estimate_tokens_from_model_flops(df, wall_time_s, model_params=model_params)
    return est["estimate_note"]


def compute_metric(trace_json_path: str, metric_type: str = "estimated_throughput_tokens_per_s", model_params: float | None = None) -> float | str:
    if metric_type == "flops_per_token_used":
        return compute_flops_per_token_used(trace_json_path, model_params)
    elif metric_type == "estimated_total_tokens":
        return compute_estimated_total_tokens(trace_json_path, model_params)
    elif metric_type == "estimated_throughput_tokens_per_s":
        return compute_estimated_throughput_tokens_per_s(trace_json_path, model_params)
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python throughput.py <trace_json_path> [metric_type] [model_params]", file=sys.stderr)
        print("  metric_type: flops_per_token_used, estimated_total_tokens, estimated_throughput_tokens_per_s, throughput_estimate_note", file=sys.stderr)
        sys.exit(1)
    
    trace_path = sys.argv[1]
    metric_type = sys.argv[2] if len(sys.argv) > 2 else "estimated_throughput_tokens_per_s"
    model_params = float(sys.argv[3]) if len(sys.argv) > 3 else None
    result = compute_metric(trace_path, metric_type, model_params)
    print(result)

