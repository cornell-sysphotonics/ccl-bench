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
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            for i, (num_tokens, loads) in enumerate(
                zip(data['num_tokens'], data['expert_loads'])
            ):
                if num_tokens > 1000:
                    continue
                total_expert_loads += np.array(loads)
                total_tokens += num_tokens
                
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    
    if total_tokens == 0:
        print("Error: No valid data found")
        return None
    
    mean_load = total_expert_loads.mean()
    std_load = total_expert_loads.std()
    cv = std_load / (mean_load + 1e-10)
    max_min_ratio = total_expert_loads.max() / (total_expert_loads.min() + 1e-10)
    
    print(f"Total tokens processed: {total_tokens}")
    print(f"Expert loads: {total_expert_loads.astype(int).tolist()}")
    print(f"Load Imbalance (CV): {cv:.4f}")
    print(f"Max/Min Ratio: {max_min_ratio:.2f}")
    
    return cv


if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1]
    compute_load_imbalance(log_dir)