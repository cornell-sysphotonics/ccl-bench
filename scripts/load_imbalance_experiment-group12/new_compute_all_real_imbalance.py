import json
import numpy as np
from pathlib import Path


def compute_load_imbalance(log_dir):
    log_dir = Path(log_dir)
    log_files = sorted(log_dir.glob("gate_logs_*.json"))
    
    total_expert_loads = np.zeros(8)
    total_tokens = 0
    
    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                data = json.load(f)
            
            for num_tokens, loads in zip(
                data["num_tokens"], data["expert_loads"]
            ):
                if num_tokens > 1000:
                    continue
                total_expert_loads += np.array(loads)
                total_tokens += num_tokens
                
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    
    if total_tokens == 0:
        return None
    
    mean_load = total_expert_loads.mean()
    std_load = total_expert_loads.std()
    cv = float(std_load / (mean_load + 1e-10))
    
    return cv


def compute_all_real_imbalance(root_dir, output_json):
    """
    Traverse a big directory and compute real imbalance (CV)
    for each subfolder.
    """
    root_dir = Path(root_dir)
    results = {}

    for subdir in sorted(root_dir.iterdir()):
        if not subdir.is_dir():
            continue

        cv = compute_load_imbalance(subdir)
        if cv is not None:
            results[subdir.name] = cv
            print(f"[OK] {subdir.name}: CV = {cv:.4f}")
        else:
            print(f"[SKIP] {subdir.name}: no valid gate logs")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved results to {output_json}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python compute_all_real_imbalance.py <ROOT_DIR> [output.json]")
        exit(1)

    root_dir = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else "real_imbalance_results.json"

    compute_all_real_imbalance(root_dir, output_json)