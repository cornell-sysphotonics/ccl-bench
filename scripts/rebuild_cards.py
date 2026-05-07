#!/usr/bin/env python3
"""
Rebuild all model and deployment cards from scratch.
- 4 models (architecture-agnostic) in trace_collection/
- 16 deployment cards (4 models × 2 phases × 2 architectures) in deployment/
"""

import yaml
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TC = REPO / "trace_collection"
DEPLOY = REPO / "deployment"


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)
    print(f"  CREATED: {path.relative_to(REPO)}")


# ── Step 1: Remove old trace_collection YAMLs ────────────────────────────────

OLD_DIRS = [
    "deepseek-v2-lite-vllm-dp_1_tp_4_ep_4-default-noCUgraph-perlmutter-group_9",
    "deepseek-v2-lite-vllm-dp_1_tp_4_ep_4-naive-noCUgraph-perlmutter-group_9",
    "deepseek-v2-lite-vllm-dp_2_tp_2_ep_4-default-noCUgraph-perlmutter-group_9",
    "deepseek-v2-lite-vllm-dp_2_tp_2_ep_4-naive-noCUgraph-perlmutter-group_9",
    "deepseek-v2-lite-vllm-dp_4_tp_1_ep_4-default-noCUgraph-perlmutter-group_9",
    "deepseek-v2-lite-vllm-dp_4_tp_1_ep_4-default-perlmutter-group_9",
    "deepseek-v2-lite-vllm-dp_4_tp_1_ep_4-naive-noCUgraph-perlmutter-group_9",
    "deepseek-v2-lite-vllm-dp_4_tp_1_ep_4-naive-perlmutter-group_9",
    "deepseek-v2-lite-vllm-dp_4_tp_1_ep_4-pplx-perlmutter-group_9",
    "deepseek-v3-16b-torchtitan-ep32-dp8-pp4-tp4-perlmutter",
    "deepseek-v3-16b-torchtitan-ep4-dp2-pp2-tp4-perlmutter",
    "deepseek-v3-16b-torchtitan-ep4-dp2-tp4-perlmutter",
    "deepseek-v3-16b-torchtitan-ep4-dp4-tp2-perlmutter",
    "deepseek-v3-16b-torchtitan-ep8-dp2-pp2-tp4-perlmutter",
    "deepseek-v3-16b-torchtitan-ep8-dp2-tp4-perlmutter",
    "deepseek-v3-16b-torchtitan-ep8-dp8-perlmutter",
    "deepseek-v3-236b-torchtitan-ep32-dp8-pp8-tp4-perlmutter",
    "llama_3.1_8b-megatron_lm-dp_2-tp_4-perlmutter-group_1",
    "llama_3.1_8b-megatron_lm-dp_4-tp_2-perlmutter-group_1",
    "Llama-3.1-8B-torchxla-vllm-tp1-tpu-group-4",
    "Llama-3.1-8B-torchxla-vllm-tp2-tpu-group-4",
    "Llama-3.1-8B-torchxla-vllm-tp4-tpu-group-4",
    "Llama-3.1-8B-torchxla-vllm-tp8-tpu-group-4",
    "Qwen3-4B-torchxla-vllm-tp1-tpu-group-4",
    "Qwen3-4B-torchxla-vllm-tp2-tpu-group-4",
    "Qwen3-4B-torchxla-vllm-tp4-tpu-group-4",
    "Qwen3-4B-torchxla-vllm-tp8-tpu-group-4",
    "llama-3.1-8b-torchxla_fsdp_v6e-4-tpu-group_21",
    "llama-3.1-8b-torchxla_fsdp_v6e-8-tpu-group_21",
    "llama-3.1-8b-torchxla_hybrid-fsdp-tp_v6e-4-tpu-group_21",
    "llama-3.1-8b-torchxla_hybrid-fsdp-tp_v6e-8-tpu-group_21",
    "llama-3.1-8b-torchxla_tp_v6e-4-tpu-group_21",
    "llama-3.1-8b-torchxla_tp_v6e-8-tpu-group_21",
]

print("=== Removing old trace_collection directories ===")
for d in OLD_DIRS:
    p = TC / d
    if p.exists():
        shutil.rmtree(p)
        print(f"  REMOVED: trace_collection/{d}")

# Remove old deployment cards
print("\n=== Removing old deployment cards ===")
if DEPLOY.exists():
    shutil.rmtree(DEPLOY)
    print("  REMOVED: deployment/")


# ── Step 2: Create new model cards ───────────────────────────────────────────

MODELS = {
    "llama-3.1-8b": {
        "version": 1,
        "description": "Llama 3.1 8B — Meta's dense 8-billion-parameter large language model. Part of the Llama 3.1 family, widely adopted for research and production workloads.",
        "hf_url": "https://huggingface.co/meta-llama/Llama-3.1-8B",
        "workload": {
            "model": {
                "model_family": "llama-3.1",
                "moe": False,
                "precision": "bf16",
                "model_arch": {
                    "num_params": 8_030_261_248,
                    "num_layers": 32,
                    "num_heads": 32,
                    "num_kv_heads": 8,
                    "head_dim": 128,
                    "hidden_size": 4096,
                    "intermediate_size": 14336,
                    "vocab_size": 128256,
                },
            },
        },
    },
    "llama-3.1-70b": {
        "version": 1,
        "description": "Llama 3.1 70B — Meta's dense 70-billion-parameter large language model. Offers strong performance across a wide range of benchmarks.",
        "hf_url": "https://huggingface.co/meta-llama/Llama-3.1-70B",
        "workload": {
            "model": {
                "model_family": "llama-3.1",
                "moe": False,
                "precision": "bf16",
                "model_arch": {
                    "num_params": 70_553_706_496,
                    "num_layers": 80,
                    "num_heads": 64,
                    "num_kv_heads": 8,
                    "head_dim": 128,
                    "hidden_size": 8192,
                    "intermediate_size": 28672,
                    "vocab_size": 128256,
                },
            },
        },
    },
    "mixtral-8x7b": {
        "version": 1,
        "description": "Mixtral 8x7B — Mistral AI's sparse Mixture-of-Experts model with 8 experts per layer, 2 active per token. 46.7B total parameters, 12.9B active.",
        "hf_url": "https://huggingface.co/mistralai/Mixtral-8x7B-v0.1",
        "workload": {
            "model": {
                "model_family": "mixtral",
                "moe": True,
                "precision": "bf16",
                "model_arch": {
                    "num_params": 46_702_792_704,
                    "num_active_params": 12_868_124_672,
                    "num_layers": 32,
                    "num_heads": 32,
                    "num_kv_heads": 8,
                    "head_dim": 128,
                    "hidden_size": 4096,
                    "intermediate_size": 14336,
                    "vocab_size": 32000,
                    "num_experts": 8,
                    "num_experts_per_tok": 2,
                },
            },
        },
    },
    "deepseek-v3": {
        "version": 1,
        "description": "DeepSeek-V3 — DeepSeek's flagship 671B Mixture-of-Experts model with Multi-head Latent Attention (MLA) and DeepSeekMoE architecture. 256 routed experts, 1 shared expert, top-8 routing.",
        "hf_url": "https://huggingface.co/deepseek-ai/DeepSeek-V3",
        "workload": {
            "model": {
                "model_family": "deepseek-v3",
                "moe": True,
                "precision": "bf16",
                "model_arch": {
                    "num_params": 671_032_320_000,
                    "num_active_params": 36_700_000_000,
                    "num_layers": 61,
                    "num_heads": 128,
                    "num_kv_heads": 128,
                    "head_dim": 128,
                    "hidden_size": 7168,
                    "intermediate_size": 18432,
                    "vocab_size": 129280,
                    "num_experts": 256,
                    "num_shared_experts": 1,
                    "num_experts_per_tok": 8,
                },
            },
        },
    },
}

print("\n=== Creating new model cards ===")
for model_name, model_data in MODELS.items():
    path = TC / model_name / f"{model_name}.yaml"
    write_yaml(path, model_data)


# ── Step 3: Create deployment cards ──────────────────────────────────────────

# Hardware specs
GPU_A100_HARDWARE = {
    "network_topo": {
        "topology": "fat-tree",
        "bandwidth_gbps": [200, 2400],  # [scale-out (Slingshot/IB), scale-up (NVLink)]
    },
    "xpu_spec": {
        "type": "GPU",
        "model": "nvidia_a100_80gb",
    },
    "driver_version": "cuda_12.4",
}

TPU_V6_HARDWARE = {
    "network_topo": {
        "topology": "2d-torus",
        "bandwidth_gbps": [200, 6400],  # [scale-out (DCN), scale-up (ICI)]
    },
    "xpu_spec": {
        "type": "TPU",
        "model": "tpu_v6e",
    },
    "driver_version": "tpu-vm-v6e-preview",
}

# Framework specs per (arch, phase)
GPU_TRAINING_FW = {
    "framework": {
        "name": "torchtitan",
        "compiler_tool_selection": "torch.compile, triton",
    },
    "communication_library": {
        "name": "NCCL",
    },
    "protocol_selection": ["rocev2", "nvlink"],
}

GPU_INFERENCE_FW = {
    "framework": {
        "name": "vllm",
        "compiler_tool_selection": "cuda",
    },
    "communication_library": {
        "name": "NCCL",
    },
    "protocol_selection": ["rocev2", "nvlink"],
}

TPU_TRAINING_FW = {
    "framework": {
        "name": "torchxla",
        "compiler_tool_selection": "xla",
    },
    "communication_library": {
        "name": "ici",
    },
    "protocol_selection": ["rdma", "ici"],
}

TPU_INFERENCE_FW = {
    "framework": {
        "name": "vllm",
        "compiler_tool_selection": "xla",
    },
    "communication_library": {
        "name": "xla",
    },
    "protocol_selection": ["xla_collectives", "ici"],
}


def make_gpu_hw(total_count, count_per_node=4):
    hw = dict(GPU_A100_HARDWARE)
    hw["xpu_spec"] = dict(hw["xpu_spec"])
    hw["xpu_spec"]["total_count"] = total_count
    hw["xpu_spec"]["count_per_node"] = count_per_node
    return hw


def make_tpu_hw(total_count, count_per_node=8, topology=None):
    hw = dict(TPU_V6_HARDWARE)
    hw["xpu_spec"] = dict(hw["xpu_spec"])
    hw["xpu_spec"]["total_count"] = total_count
    hw["xpu_spec"]["count_per_node"] = count_per_node
    if topology:
        hw["xpu_spec"]["topology"] = topology
    return hw


# Deployment configurations: (model, phase, arch) -> deployment details
DEPLOYMENTS = [
    # ── Llama-3.1-8B ────────────────────────────────────────────────────
    {
        "model": "llama-3.1-8b",
        "phase": "training",
        "arch": "gpu-a100",
        "description": "Llama-3.1-8B training on NVIDIA A100 GPUs using torchtitan with FSDP.",
        "hardware": make_gpu_hw(total_count=8),
        "fw": GPU_TRAINING_FW,
        "data": {"batch_size": 32, "seq_len": 4096, "dataset": "c4"},
        "granularity": "model_fwd_bwd_pass",
        "parallelism": {"dp_replicate": 1, "dp_shard": 8, "tp": 1, "pp": 1, "cp": 1},
        "metric_source": {"traces": ["json"]},
    },
    {
        "model": "llama-3.1-8b",
        "phase": "inference",
        "arch": "gpu-a100",
        "description": "Llama-3.1-8B inference on NVIDIA A100 GPUs using vLLM.",
        "hardware": make_gpu_hw(total_count=1, count_per_node=1),
        "fw": GPU_INFERENCE_FW,
        "data": {"batch_size": 128, "seq_len": 4096, "dataset": "sharegpt"},
        "granularity": "model_fwd",
        "parallelism": {"dp_replicate": 1, "dp_shard": 1, "tp": 1, "pp": 1, "cp": 1},
        "metric_source": {"traces": ["json"]},
    },
    {
        "model": "llama-3.1-8b",
        "phase": "training",
        "arch": "tpu-v6",
        "description": "Llama-3.1-8B training on TPU v6e using torchxla with FSDP.",
        "hardware": make_tpu_hw(total_count=8, topology="mesh-2x2x2"),
        "fw": TPU_TRAINING_FW,
        "data": {"batch_size": 16, "seq_len": 4096, "dataset": "c4"},
        "granularity": "model_fwd_bwd_pass",
        "parallelism": {"dp_replicate": 1, "dp_shard": 8, "tp": 1, "pp": 1, "cp": 1},
        "metric_source": {"traces": ["json_tpu"]},
    },
    {
        "model": "llama-3.1-8b",
        "phase": "inference",
        "arch": "tpu-v6",
        "description": "Llama-3.1-8B inference on TPU v6e using vLLM with XLA backend.",
        "hardware": make_tpu_hw(total_count=4, topology="mesh-2x2x1"),
        "fw": TPU_INFERENCE_FW,
        "data": {"batch_size": 128, "seq_len": 4096, "dataset": "sharegpt"},
        "granularity": "model_fwd",
        "parallelism": {"dp_replicate": 1, "dp_shard": 1, "tp": 4, "pp": 1, "cp": 1},
        "metric_source": {"traces": ["json_tpu"]},
    },

    # ── Llama-3.1-70B ───────────────────────────────────────────────────
    {
        "model": "llama-3.1-70b",
        "phase": "training",
        "arch": "gpu-a100",
        "description": "Llama-3.1-70B training on NVIDIA A100 GPUs using torchtitan with FSDP + TP.",
        "hardware": make_gpu_hw(total_count=32),
        "fw": GPU_TRAINING_FW,
        "data": {"batch_size": 16, "seq_len": 4096, "dataset": "c4"},
        "granularity": "model_fwd_bwd_pass",
        "parallelism": {"dp_replicate": 1, "dp_shard": 8, "tp": 4, "pp": 1, "cp": 1},
        "metric_source": {"traces": ["json"]},
    },
    {
        "model": "llama-3.1-70b",
        "phase": "inference",
        "arch": "gpu-a100",
        "description": "Llama-3.1-70B inference on NVIDIA A100 GPUs using vLLM with TP=4.",
        "hardware": make_gpu_hw(total_count=4),
        "fw": GPU_INFERENCE_FW,
        "data": {"batch_size": 64, "seq_len": 4096, "dataset": "sharegpt"},
        "granularity": "model_fwd",
        "parallelism": {"dp_replicate": 1, "dp_shard": 1, "tp": 4, "pp": 1, "cp": 1},
        "metric_source": {"traces": ["json"]},
    },
    {
        "model": "llama-3.1-70b",
        "phase": "training",
        "arch": "tpu-v6",
        "description": "Llama-3.1-70B training on TPU v6e using torchxla with FSDP + TP.",
        "hardware": make_tpu_hw(total_count=32, topology="mesh-4x4x2"),
        "fw": TPU_TRAINING_FW,
        "data": {"batch_size": 8, "seq_len": 4096, "dataset": "c4"},
        "granularity": "model_fwd_bwd_pass",
        "parallelism": {"dp_replicate": 1, "dp_shard": 8, "tp": 4, "pp": 1, "cp": 1},
        "metric_source": {"traces": ["json_tpu"]},
    },
    {
        "model": "llama-3.1-70b",
        "phase": "inference",
        "arch": "tpu-v6",
        "description": "Llama-3.1-70B inference on TPU v6e using vLLM with XLA backend and TP=8.",
        "hardware": make_tpu_hw(total_count=8, topology="mesh-2x2x2"),
        "fw": TPU_INFERENCE_FW,
        "data": {"batch_size": 64, "seq_len": 4096, "dataset": "sharegpt"},
        "granularity": "model_fwd",
        "parallelism": {"dp_replicate": 1, "dp_shard": 1, "tp": 8, "pp": 1, "cp": 1},
        "metric_source": {"traces": ["json_tpu"]},
    },

    # ── Mixtral-8x7B ────────────────────────────────────────────────────
    {
        "model": "mixtral-8x7b",
        "phase": "training",
        "arch": "gpu-a100",
        "description": "Mixtral-8x7B MoE training on NVIDIA A100 GPUs using torchtitan with FSDP + EP.",
        "hardware": make_gpu_hw(total_count=16),
        "fw": GPU_TRAINING_FW,
        "data": {"batch_size": 16, "seq_len": 4096, "dataset": "c4"},
        "granularity": "model_fwd_bwd_pass",
        "parallelism": {"dp_replicate": 1, "dp_shard": 2, "tp": 1, "pp": 1, "cp": 1, "ep": 8},
        "metric_source": {"traces": ["json"]},
    },
    {
        "model": "mixtral-8x7b",
        "phase": "inference",
        "arch": "gpu-a100",
        "description": "Mixtral-8x7B MoE inference on NVIDIA A100 GPUs using vLLM with TP=4.",
        "hardware": make_gpu_hw(total_count=4),
        "fw": GPU_INFERENCE_FW,
        "data": {"batch_size": 64, "seq_len": 4096, "dataset": "sharegpt"},
        "granularity": "model_fwd",
        "parallelism": {"dp_replicate": 1, "dp_shard": 1, "tp": 4, "pp": 1, "cp": 1, "ep": 1},
        "metric_source": {"traces": ["json"]},
    },
    {
        "model": "mixtral-8x7b",
        "phase": "training",
        "arch": "tpu-v6",
        "description": "Mixtral-8x7B MoE training on TPU v6e using torchxla with FSDP + EP.",
        "hardware": make_tpu_hw(total_count=16, topology="mesh-4x2x2"),
        "fw": TPU_TRAINING_FW,
        "data": {"batch_size": 8, "seq_len": 4096, "dataset": "c4"},
        "granularity": "model_fwd_bwd_pass",
        "parallelism": {"dp_replicate": 1, "dp_shard": 2, "tp": 1, "pp": 1, "cp": 1, "ep": 8},
        "metric_source": {"traces": ["json_tpu"]},
    },
    {
        "model": "mixtral-8x7b",
        "phase": "inference",
        "arch": "tpu-v6",
        "description": "Mixtral-8x7B MoE inference on TPU v6e using vLLM with XLA backend and TP=4.",
        "hardware": make_tpu_hw(total_count=8, topology="mesh-2x2x2"),
        "fw": TPU_INFERENCE_FW,
        "data": {"batch_size": 64, "seq_len": 4096, "dataset": "sharegpt"},
        "granularity": "model_fwd",
        "parallelism": {"dp_replicate": 1, "dp_shard": 1, "tp": 4, "pp": 1, "cp": 1, "ep": 2},
        "metric_source": {"traces": ["json_tpu"]},
    },

    # ── DeepSeek-V3 ─────────────────────────────────────────────────────
    {
        "model": "deepseek-v3",
        "phase": "training",
        "arch": "gpu-a100",
        "description": "DeepSeek-V3 671B MoE training on NVIDIA A100 GPUs using torchtitan with 3D parallelism + EP.",
        "hardware": make_gpu_hw(total_count=256),
        "fw": GPU_TRAINING_FW,
        "data": {"batch_size": 64, "seq_len": 4096, "dataset": "c4"},
        "granularity": "model_fwd_bwd_pass",
        "parallelism": {"dp_replicate": 1, "dp_shard": 8, "tp": 4, "pp": 8, "cp": 1, "ep": 32},
        "metric_source": {"traces": ["json"]},
    },
    {
        "model": "deepseek-v3",
        "phase": "inference",
        "arch": "gpu-a100",
        "description": "DeepSeek-V3 671B MoE inference on NVIDIA A100 GPUs using vLLM with TP + EP.",
        "hardware": make_gpu_hw(total_count=32),
        "fw": GPU_INFERENCE_FW,
        "data": {"batch_size": 32, "seq_len": 4096, "dataset": "sharegpt"},
        "granularity": "model_fwd",
        "parallelism": {"dp_replicate": 1, "dp_shard": 1, "tp": 4, "pp": 1, "cp": 1, "ep": 8},
        "metric_source": {"traces": ["json"]},
    },
    {
        "model": "deepseek-v3",
        "phase": "training",
        "arch": "tpu-v6",
        "description": "DeepSeek-V3 671B MoE training on TPU v6e using torchxla with 3D parallelism + EP.",
        "hardware": make_tpu_hw(total_count=256, topology="mesh-8x8x4"),
        "fw": TPU_TRAINING_FW,
        "data": {"batch_size": 64, "seq_len": 4096, "dataset": "c4"},
        "granularity": "model_fwd_bwd_pass",
        "parallelism": {"dp_replicate": 1, "dp_shard": 8, "tp": 4, "pp": 8, "cp": 1, "ep": 32},
        "metric_source": {"traces": ["json_tpu"]},
    },
    {
        "model": "deepseek-v3",
        "phase": "inference",
        "arch": "tpu-v6",
        "description": "DeepSeek-V3 671B MoE inference on TPU v6e using vLLM with XLA backend, TP + EP.",
        "hardware": make_tpu_hw(total_count=32, topology="mesh-4x4x2"),
        "fw": TPU_INFERENCE_FW,
        "data": {"batch_size": 32, "seq_len": 4096, "dataset": "sharegpt"},
        "granularity": "model_fwd",
        "parallelism": {"dp_replicate": 1, "dp_shard": 1, "tp": 4, "pp": 1, "cp": 1, "ep": 8},
        "metric_source": {"traces": ["json_tpu"]},
    },
]

print("\n=== Creating deployment cards ===")
for dep in DEPLOYMENTS:
    model = dep["model"]
    phase = dep["phase"]
    arch = dep["arch"]

    fw_spec = dep["fw"]

    card = {
        "version": 1,
        "model_card": f"trace_collection/{model}/{model}.yaml",
        "description": dep["description"],
        "workload": {
            "phase": phase,
            "granularity": dep["granularity"],
            "data": dep["data"],
        },
        "hardware": dep["hardware"],
        "Model-executor": {
            "framework": fw_spec["framework"],
            "model_plan_parallelization": dep["parallelism"],
            "communication_library": {"name": fw_spec["communication_library"]["name"]},
            "protocol_selection": fw_spec["protocol_selection"],
        },
        "metric_source": dep["metric_source"],
    }

    filename = f"{model}-{phase}.yaml"
    path = DEPLOY / arch / filename
    write_yaml(path, card)

print(f"\nDone: {len(MODELS)} model cards, {len(DEPLOYMENTS)} deployment cards")
