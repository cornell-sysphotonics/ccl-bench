#!/usr/bin/env python3
"""
Generate all experiment configuration files for E1.1-E3.4
"""

import os
import yaml


EXPERIMENTS = {
    # TP scaling for Llama-8B (inference)
    "E1.1": {
        "description": "Llama-8B baseline on 1 GPU (TP=1, PP=1, EP=1)",
        "model": "meta-llama/Llama-3.1-8B",
        "parallelism": {"tp": 1, "pp": 1, "dp_replicate": 1, "dp_shard": 1, "ep": 1},
        "gpus": 1,
        "batch_size": 4,
    },
    "E1.2": {
        "description": "Llama-8B tensor parallelism TP=2 on 2 GPUs",
        "model": "meta-llama/Llama-3.1-8B",
        "parallelism": {"tp": 2, "pp": 1, "dp_replicate": 1, "dp_shard": 1, "ep": 1},
        "gpus": 2,
        "batch_size": 4,
    },
    "E1.3": {
        "description": "Llama-8B tensor parallelism TP=4 on 4 GPUs",
        "model": "meta-llama/Llama-3.1-8B",
        "parallelism": {"tp": 4, "pp": 1, "dp_replicate": 1, "dp_shard": 1, "ep": 1},
        "gpus": 4,
        "batch_size": 4,
    },
}


def generate_config(exp_id, exp_info):
    """Generate configuration file for an experiment."""
    config = {
        "experiment_id": exp_id,
        "description": exp_info["description"],
        "model": {
            "name": exp_info["model"],
            "precision": "bfloat16",
        },
        "parallelism": exp_info["parallelism"],
        "data": {
            "batch_size": exp_info["batch_size"],
            "seq_len": 2048,
            "max_tokens": 512,
        },
        "warmup_iterations": 2,
        "profile_iterations": 5,
        "output_dir": f"trace_collection/llama-8b-tp{exp_info['parallelism']['tp']}",
    }

    return config


def main():
    os.makedirs("experiments/configs", exist_ok=True)

    for exp_id, exp_info in EXPERIMENTS.items():
        config = generate_config(exp_id, exp_info)

        filename = f"experiments/configs/{exp_id}_llama8b_tp{exp_info['parallelism']['tp']}.yaml"

        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Generated {filename}")

    print(f"\nTotal experiments generated: {len(EXPERIMENTS)}")


if __name__ == "__main__":
    main()
