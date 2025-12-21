#!/usr/bin/env python3
"""
Generate workload card from experiment configuration
"""

import argparse
import yaml
import os


def generate_workload_card(config_path, output_dir):
    """Generate a workload card YAML from experiment configuration."""

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    exp_id = config['experiment_id']
    model_name = config['model']['name'].split('/')[-1]

    # Determine if MoE model
    is_moe = 'deepseek' in model_name.lower()

    workload_card = {
        'version': 1,
        'description': config['description'],
        'hf_url': f"https://huggingface.co/{config['model']['name']}",
        'trace_url': '',  # To be filled after uploading traces
        'workload': {
            'model': {
                'phase': 'inference',
                'moe': is_moe,
                'granularity': 'model_fwd',
                'model_family': model_name.lower(),
                'precision': config['model']['precision'],
            },
            'data': {
                'batch_size': config['data']['batch_size'],
                'seq_len': config['data']['seq_len'],
                'dataset': 'synthetic',
            },
            'hardware': {
                'network_topo': {
                    'topology': 'slingshot',
                    'bandwidth_gbps': [200, 2000],  # Scale-out, Scale-up
                },
                'xpu_spec': {
                    'type': 'GPU',
                    'model': 'nvidia_a100',
                    'total_count': sum(config['parallelism'].values()),
                    'count_per_node': 4,
                },
                'driver_version': 'cuda_12.4',
            },
        },
        'Model-executor': {
            'framework': {
                'name': 'vllm',
                'compiler_tool_selection': 'plain_pytorch',
            },
            'model_plan_parallelization': {
                'dp_replicate': config['parallelism']['dp_replicate'],
                'dp_shard': config['parallelism']['dp_shard'],
                'tp': config['parallelism']['tp'],
                'pp': config['parallelism']['pp'],
                'cp': 1,
                'ep': config['parallelism'].get('ep', 1),
            },
            'communication_library': {
                'name': 'NCCL',
                'version': '2.18+',
                'env': {
                    'NCCL_IB_QPS_PER_CONNECTION': None,
                },
            },
            'protocol_selection': ['rocev2', 'p2p'],
        },
        'metric_source': {
            'traces': ['torch_et', 'kineto_trace', 'nsys'],
            'metrics_specific_trace': [],
        },
    }

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write workload card
    output_path = os.path.join(output_dir, 'workload_card.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(workload_card, f, default_flow_style=False, sort_keys=False)

    print(f"Generated workload card: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate workload card from experiment config")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment configuration YAML")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for workload card")

    args = parser.parse_args()

    generate_workload_card(args.config, args.output_dir)


if __name__ == "__main__":
    main()
