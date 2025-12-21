import json
import numpy as np
from pathlib import Path

def compute_load_imbalance(log_dir):
    """
    Compute load imbalance score from MoE gate logs.
    
    Args:
        log_dir: Path to directory containing gate_logs_*.json files
    
    Returns:
        float: Coefficient of Variation (CV) - lower is better (0 = perfect balance)
    """
    log_dir = Path(log_dir)
    log_files = sorted(log_dir.glob("gate_logs_*.json"))
    
    all_expert_loads = []
    files_processed = 0
    files_skipped = 0
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            # Skip warmup/profiling steps
            for num_tokens, loads in zip(data['num_tokens'], data['expert_loads']):
                all_expert_loads.append(loads)
            
            files_processed += 1
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            files_skipped += 1
            continue
    
    if not all_expert_loads:
        print("Error: No valid data found")
        return None
    
    all_expert_loads = np.array(all_expert_loads)
    
    # Compute CV per step, then average
    step_means = all_expert_loads.mean(axis=1)
    step_stds = all_expert_loads.std(axis=1)
    step_cvs = step_stds / (step_means + 1e-10)
    
    cv = float(np.mean(step_cvs))
    
    print(f"Files: {files_processed} processed, {files_skipped} skipped")
    print(f"Load Imbalance Score (CV): {cv:.4f}")
    print(f"  (Lower is better: <0.1=excellent, 0.1-0.3=good, 0.3-0.5=moderate, >0.5=severe)")
    
    return cv


if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "/pscratch/sd/y/yz3526/final_project/vllm_gate_logs"
    score = compute_load_imbalance(log_dir)