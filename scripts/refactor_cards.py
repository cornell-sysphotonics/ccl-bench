#!/usr/bin/env python3
"""
Refactor workload cards:
1. Strip hardware/framework/deployment fields from trace_collection/ YAML files
2. Create deployment/ cards with full hardware+framework specs
"""

import yaml
import os
import copy
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TC = REPO / "trace_collection"
DEPLOY = REPO / "deployment"

# Map each workload dir to its deployment target architecture folder
ARCH_MAP = {}

# GPU A100 workloads
for name in [
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
]:
    ARCH_MAP[name] = "gpu-a100"

# TPU v6e workloads
for name in [
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
]:
    ARCH_MAP[name] = "tpu-v6"


# Framework selection: one framework per (architecture, workload_type)
# gpu-a100 + training → torchtitan (for deepseek), megatron-lm (for llama)
# gpu-a100 + serving  → vllm
# tpu-v6   + inference → vllm (torchxla backend)
# tpu-v6   + training  → torchxla


def strip_deployment_name(name):
    """Create a cleaner deployment card name by removing system-specific suffixes."""
    # Remove -perlmutter, -group_N, -group-N suffixes
    import re
    clean = re.sub(r'-perlmutter', '', name)
    clean = re.sub(r'-group[_-]\d+', '', clean)
    # Remove trailing dashes
    clean = clean.strip('-')
    return clean


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)


def make_model_description(data):
    """Create a hardware-agnostic description from the original."""
    w = data.get("workload", {}) or {}
    mod = w.get("model", {}) or {}
    family = mod.get("model_family", "")
    phase = mod.get("phase", "")
    moe = mod.get("moe", False)

    parts = []
    if family:
        parts.append(family)
    if moe:
        parts.append("Mixture-of-Experts")
    if phase:
        parts.append(f"model for {phase}")
    else:
        parts.append("model")

    return " ".join(parts) + "."


def strip_workload_card(data):
    """Return a model-only workload card (no hardware/framework/deployment fields)."""
    stripped = {}

    # Keep version
    if "version" in data:
        stripped["version"] = data["version"]

    # Rewrite description to be model-agnostic
    stripped["description"] = make_model_description(data)

    # Keep hf_url
    if "hf_url" in data:
        stripped["hf_url"] = data.get("hf_url")

    # Keep workload.model and workload.data only
    w = data.get("workload", {}) or {}
    stripped_workload = {}
    if "model" in w:
        stripped_workload["model"] = copy.deepcopy(w["model"])
    if "data" in w:
        stripped_workload["data"] = copy.deepcopy(w["data"])
    stripped["workload"] = stripped_workload

    return stripped


def make_deployment_card(data, workload_dir_name):
    """Create a deployment card with full hardware+framework specs."""
    deploy = {}

    deploy["version"] = data.get("version", 1)

    # Reference back to the model card
    deploy["model_card"] = f"trace_collection/{workload_dir_name}/{workload_dir_name}.yaml"

    # Keep original description (deployment-specific)
    desc = data.get("description", "")
    if isinstance(desc, str):
        deploy["description"] = desc.strip()
    else:
        deploy["description"] = str(desc)

    # trace_url
    if "trace_url" in data:
        deploy["trace_url"] = data.get("trace_url")

    # Hardware section
    w = data.get("workload", {}) or {}
    if "hardware" in w:
        deploy["hardware"] = copy.deepcopy(w["hardware"])

    # Model-executor section
    if "Model-executor" in data:
        deploy["Model-executor"] = copy.deepcopy(data["Model-executor"])

    # metric_source section
    if "metric_source" in data:
        deploy["metric_source"] = copy.deepcopy(data["metric_source"])

    return deploy


def main():
    # Create deployment directories
    (DEPLOY / "gpu-a100").mkdir(parents=True, exist_ok=True)
    (DEPLOY / "tpu-v6").mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for workload_dir in sorted(TC.iterdir()):
        if not workload_dir.is_dir():
            continue

        name = workload_dir.name
        yaml_files = sorted(workload_dir.glob("*.yaml"))
        if not yaml_files:
            print(f"  SKIP (no YAML): {name}")
            skipped += 1
            continue

        yaml_path = yaml_files[0]
        data = load_yaml(yaml_path)
        if not data:
            print(f"  SKIP (empty YAML): {name}")
            skipped += 1
            continue

        arch = ARCH_MAP.get(name)
        if not arch:
            print(f"  SKIP (not in ARCH_MAP): {name}")
            skipped += 1
            continue

        # 1. Create deployment card
        deploy_card = make_deployment_card(data, name)
        deploy_name = strip_deployment_name(name)
        deploy_path = DEPLOY / arch / f"{deploy_name}.yaml"
        write_yaml(deploy_path, deploy_card)
        print(f"  DEPLOY: {deploy_path.relative_to(REPO)}")

        # 2. Strip workload card in place
        stripped = strip_workload_card(data)
        write_yaml(yaml_path, stripped)
        print(f"  STRIP:  {yaml_path.relative_to(REPO)}")

        processed += 1

    print(f"\nDone: {processed} processed, {skipped} skipped")


if __name__ == "__main__":
    main()
