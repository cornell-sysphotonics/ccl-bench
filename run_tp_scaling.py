#!/usr/bin/env python3
"""
Run TP scaling experiments (E1.1, E1.2, E1.3) and aggregate metrics.
"""

import os
import subprocess
import sys


EXPERIMENT_CONFIGS = [
    "experiments/configs/E1.1_llama8b_baseline.yaml",
    "experiments/configs/E1.2_llama8b_tp2.yaml",
    "experiments/configs/E1.3_llama8b_tp4.yaml",
]


def run_command(cmd: str) -> None:
    print(f"Running: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] Command failed ({exc.returncode}): {cmd}")


def main():
    print("=== TP Scaling (Llama-8B, TP=1/2/4) ===")

    for config in EXPERIMENT_CONFIGS:
        if not os.path.exists(config):
            print(f"[ERROR] Config not found: {config}")
            continue
        print(f"\n--- Experiment: {config} ---")
        run_command(f"python vllm_profiler.py --config {config}")

    print("\n=== Aggregating metrics ===")
    run_command("python experiments/analyze_results.py")

    print("\nDone. See experiments/results_summary.csv and experiments/scaling_analysis.png")


if __name__ == "__main__":
    main()
