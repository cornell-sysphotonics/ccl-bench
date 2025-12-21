# Extracts token_to_expert_assignment metric from gate logs - returns JSON with totals, normalized distribution, and summary statistics
import json
import sys
import math
from pathlib import Path
from collections import defaultdict

def extract_token_to_expert_assignment(gates_logs_dir):
    gates_logs_dir = Path(gates_logs_dir)
    #aggregate tokens per expert across all gate log subdirectories
    expert_totals = defaultdict(float)
    total_steps = 0
    for log_subdir in sorted(gates_logs_dir.glob("gates_logs_*")):
        if not log_subdir.is_dir():
            continue
        
        for log_file in sorted(log_subdir.glob("gate_logs_*.json")):
            try:
                with open(log_file) as f:
                    data = json.load(f)
                
                if "expert_loads" not in data:
                    continue
                
                # Aggregate tokens assigned to each expert across all steps in this file
                for expert_loads in data["expert_loads"]:
                    if not expert_loads:
                        continue
                    for expert_id, load in enumerate(expert_loads):
                        expert_totals[expert_id] += load
                    total_steps += 1
                    
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    
    if not expert_totals:
        return None
    
    num_experts = max(expert_totals.keys()) + 1
    totals = [expert_totals.get(i, 0.0) for i in range(num_experts)]
    total_tokens = sum(totals)
    
    if total_tokens == 0:
        return None
    
    # Normalized distribution (proportions)
    normalized = [t / total_tokens for t in totals]
    
    # Summary statistics
    # Entropy: measures diversity of assignments (higher = more uniform)
    entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in normalized)
    max_entropy = math.log2(num_experts)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Coefficient of Variation (CV): measures relative variability (lower = more balanced)
    mean_load = sum(totals) / num_experts
    variance = sum((t - mean_load) ** 2 for t in totals) / num_experts
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_load if mean_load > 0 else float('inf')
    
    # Max/Min ratio: imbalance ratio
    non_zero_loads = [t for t in totals if t > 0]
    if non_zero_loads:
        max_load = max(totals)
        min_load = min(non_zero_loads) if non_zero_loads else 0
        max_min_ratio = max_load / min_load if min_load > 0 else float('inf')
    else:
        max_min_ratio = 0.0
    
    return {
        "totals": totals,
        "normalized": normalized,
        "summary": {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "coefficient_of_variation": cv,
            "max_min_ratio": max_min_ratio,
            "total_tokens": total_tokens,
            "num_experts": num_experts,
            "total_steps": total_steps
        }
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        trace_dir = Path(sys.argv[1])
        
        # Check if the trace_dir itself is a gates_logs directory (contains gates_logs_* subdirectories)
        if list(trace_dir.glob("gates_logs_*")):
            gates_logs_dir = trace_dir
        else:
            # Otherwise, look for gates_logs directories within trace_dir
            gates_logs_dirs = list(trace_dir.glob("*gates_logs*"))
            if gates_logs_dirs:
                gates_logs_dir = gates_logs_dirs[0]
            else:
                gates_logs_dir = trace_dir / "baseline_gates_logs"
    else:
        gates_logs_dir = Path("baseline_gates_logs")
    
    if not gates_logs_dir.exists():
        error_msg = f"Gates logs directory not found: {gates_logs_dir}. Expected directory matching '*gates_logs*' or a directory containing 'gates_logs_*' subdirectories."
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)
    
    if not gates_logs_dir.is_dir():
        error_msg = f"Path exists but is not a directory: {gates_logs_dir}"
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)
    
    result = extract_token_to_expert_assignment(gates_logs_dir)
    if result:
        print(json.dumps(result, indent=2))
    else:
        error_msg = f"No gate log data found in {gates_logs_dir}. Ensure the directory contains 'gates_logs_*' subdirectories with 'gate_logs_*.json' files containing 'expert_loads' data."
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)

