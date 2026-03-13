"""
Compute Hockney model parameters for AllReduce
"""
import sys
import os
import importlib.util
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import yaml

# Import utils-group-21
tools_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_path = os.path.join(tools_dir, "utils-group-21.py")
spec = importlib.util.spec_from_file_location("utils_group_21", utils_path)
utils_group_21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_group_21)
prepare_dataframe = utils_group_21.prepare_dataframe

def get_n_chips_from_card(trace_json_path):
    base = trace_json_path
    for ext in (".trace.json", ".json.gz", ".json"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break

    yaml_path = base + ".yaml"

    if not os.path.exists(yaml_path):
        folder = os.path.dirname(trace_json_path)
        yaml_files = [f for f in os.listdir(folder) if f.endswith('.yaml')]
        if not yaml_files:
            return None
        yaml_path = os.path.join(folder, yaml_files[0])

    with open(yaml_path, "r") as f:
        card = yaml.safe_load(f)
    try:
        return card["workload"]["hardware"]["xpu_spec"]["total_count"]
    except Exception as e:
        return None
    
def fit_hockney_allreduce(df: pd.DataFrame, n_chips: int) -> tuple[float, float, float, str]:
    comm_df = df[df["kind"] == "comm"].copy()
    bw_df = comm_df.dropna(subset=["message_bytes", "T_s"]).copy()
    bw_df = bw_df[(bw_df["message_bytes"] > 0) & (bw_df["T_s"] > 0)]
    ar_df = bw_df[bw_df["name_l"].str.contains("allreduce|all-reduce|all_reduce", regex=True, na=False)].copy()
    
    alpha = beta = np.nan
    note = ""

    if len(ar_df) >= 5 and ar_df["message_bytes"].nunique() >= 2:
        x = (ar_df["message_bytes"] / n_chips).to_numpy().reshape(-1, 1)  # S/n
        y = (ar_df["T_s"] / (2 * (n_chips - 1))).to_numpy()  # T/(2(n-1))
        
        reg = LinearRegression().fit(x, y)
        alpha = float(reg.intercept_)
        beta = float(reg.coef_[0])
        note = "No problems fitting Hockney Model"
    else:
        if len(ar_df) < 5:
            note = "Not enough AllReduce samples to fit (need >= 5)."
        elif ar_df["message_bytes"].nunique() < 2:
            note = "AllReduce message size is constant; alpha/beta are not identifiable."

    inv_beta = (1.0 / beta) if np.isfinite(beta) and beta > 0 else np.nan
    return alpha, beta, float(inv_beta) if np.isfinite(inv_beta) else np.nan, note


def compute_alpha(trace_json_path: str, n_chips: int) -> float:
    df = prepare_dataframe(trace_json_path)
    alpha, _, _, _ = fit_hockney_allreduce(df, n_chips)
    return alpha


def compute_beta(trace_json_path: str, n_chips: int) -> float:
    df = prepare_dataframe(trace_json_path)
    _, beta, _, _ = fit_hockney_allreduce(df, n_chips)
    return beta


def compute_inverse_beta(trace_json_path: str, n_chips: int) -> float:
    df = prepare_dataframe(trace_json_path)
    _, _, inv_beta, _ = fit_hockney_allreduce(df, n_chips)
    return inv_beta


def compute_note(trace_json_path: str, n_chips: int) -> str:
    df = prepare_dataframe(trace_json_path)
    _, _, _, note = fit_hockney_allreduce(df, n_chips)
    return note


def compute_metric(trace_json_path: str, metric_type: str = "alpha") -> float | str:
    """
    User need to specify metric to get either alpha, beta, inverse_beta
    """
    n_chips_card = get_n_chips_from_card(trace_json_path)
    if n_chips_card is None:
        raise ValueError("n_chips not found in workload card YAML")
    if metric_type == "alpha":
        return compute_alpha(trace_json_path, n_chips_card)
    elif metric_type == "beta":
        return compute_beta(trace_json_path, n_chips_card)
    elif metric_type == "inverse_beta":
        return compute_inverse_beta(trace_json_path, n_chips_card)
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hockney.py <trace_json_path> [alpha|beta|inverse_beta|note]", file=sys.stderr)
        sys.exit(1)

    trace_path = sys.argv[1]
    metric_type = sys.argv[2] if len(sys.argv) > 2 else "alpha"
    result = compute_metric(trace_path, metric_type=metric_type)
    print(result)

