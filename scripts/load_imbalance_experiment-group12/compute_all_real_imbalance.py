import json
import numpy as np
from pathlib import Path


def compute_load_imbalance(log_dir):
    """
    Compute real MoE routing load imbalance (CV) for one folder.
    """
    log_dir = Path(log_dir)
    log_files = sorted(log_dir.glob("gate_logs_*.json"))

    all_expert_loads = []

    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                data = json.load(f)

            for loads in data.get("expert_loads", []):
                all_expert_loads.append(loads)

        except Exception:
            continue

    if not all_expert_loads:
        return None

    all_expert_loads = np.array(all_expert_loads)

    step_means = all_expert_loads.mean(axis=1)
    step_stds = all_expert_loads.std(axis=1)
    step_cvs = step_stds / (step_means + 1e-10)

    return float(step_cvs.mean())


def compute_all_folders(root_dir, output_json):
    """
    Traverse all subfolders and compute imbalance per folder.
    """
    root_dir = Path(root_dir)
    results = {}

    for subfolder in sorted(root_dir.iterdir()):
        if not subfolder.is_dir():
            continue

        print(f"📂 Processing {subfolder.name} ...")
        score = compute_load_imbalance(subfolder)

        if score is None:
            print("  ⚠️  No valid gate logs found")
            continue

        results[subfolder.name] = score
        print(f"  ✓ CV = {score:.4f}")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved results to {output_json}")


if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "./baseline_gates_logs"
    output_json = sys.argv[2] if len(sys.argv) > 2 else "real_imbalance_results.json"

    compute_all_folders(root_dir, output_json)