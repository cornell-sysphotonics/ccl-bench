#!/usr/bin/env python3
"""
Task B: Scheduler & Latency Expert
Execution script for experiments E2.1, E2.2, and E2.3.
"""

import os
import subprocess
import sys
import time

def run_command(cmd):
    print(f"Running: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Exit code: {e.returncode}")
        # Continue even if one fails, to try others
        pass

def main():
    print("=== Starting Task B Experiments (Scheduler & Latency) ===")
    
    # Define experiments
    experiments = [
        "experiments/configs/E2.1_qwen_tp4.yaml",
        "experiments/configs/E2.2_qwen_tp2_pp2.yaml",
        "experiments/configs/E2.3_qwen_pp4.yaml"
    ]
    
    # 1. Run Experiments
    for config in experiments:
        if not os.path.exists(config):
            print(f"Error: Config file {config} not found!")
            continue
            
        print(f"\n--- Running Experiment: {config} ---")
        # Assuming python is available in the environment where this is run
        cmd = f"python vllm_profiler.py --config {config}"
        run_command(cmd)
        
    print("\n=== Experiments Complete ===")
    
    # 2. Analyze Results
    print("\n=== Analyzing Results ===")
    cmd = "python experiments/analyze_results.py"
    run_command(cmd)
    
    print("\n=== Task B Complete ===")
    print("Please check 'experiments/results_summary.csv' for metrics:")
    print("- TTFT (ttft_ms)")
    print("- TPOT (tpot_ms)")
    print("- Pipeline Bubble Ratio (bubble_ratio_pct)")

if __name__ == "__main__":
    main()
