#!/usr/bin/env python3
"""
Replace benchmark/ workload cards with the 4 models from deployment/.
Merges model card (trace_collection/) + deployment card into the full
benchmark card format.
"""

import yaml
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "benchmark"
DEPLOY = REPO / "deployment"
TC = REPO / "trace_collection"


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)
    print(f"  CREATED: {path.relative_to(REPO)}")


def hw_tag(dep):
    """Short hardware tag for filename: a100, tpuv6e, etc."""
    xpu = dep.get("hardware", {}).get("xpu_spec", {})
    model = xpu.get("model", "")
    if "a100" in model:
        return "a100"
    if "v6e" in model:
        return "tpuv6e"
    return model.replace("nvidia_", "").replace("tpu_", "tpu")


def parallelism_tag(par):
    """Build parallelism suffix from non-trivial dims, e.g. tp4-ep8-dp2."""
    parts = []
    mapping = [("ep", "ep"), ("tp", "tp"), ("pp", "pp"), ("cp", "cp")]
    for key, label in mapping:
        v = par.get(key, 1)
        if v and v > 1:
            parts.append(f"{label}{v}")
    # dp: combine replicate + shard
    dp_r = par.get("dp_replicate", 1) or 1
    dp_s = par.get("dp_shard", 1) or 1
    if dp_s > 1:
        parts.append(f"fsdp{dp_s}")
    elif dp_r > 1:
        parts.append(f"dp{dp_r}")
    return "-".join(parts) if parts else "single"


def merge(model_card, deploy_card, model_name):
    """Merge model card + deployment card into full benchmark card."""
    mc = model_card.get("workload", {}).get("model", {})
    dep_wl = deploy_card.get("workload", {})

    card = {
        "version": 1,
        "description": deploy_card.get("description", ""),
        "hf_url": model_card.get("hf_url", ""),
        "trace_url": "",
        "contributor": "",
        "contact": "",
        "workload": {
            "model": {
                "phase": dep_wl.get("phase", ""),
                "moe": mc.get("moe", False),
                "granularity": dep_wl.get("granularity", ""),
                "model_family": mc.get("model_family", ""),
                "precision": mc.get("precision", "bf16"),
                "epochs": 1,
                "iteration": 100 if dep_wl.get("phase") == "inference" else 20,
                "model_arch": mc.get("model_arch", {}),
            },
            "data": dep_wl.get("data", {}),
            "hardware": deploy_card.get("hardware", {}),
        },
        "Model-executor": deploy_card.get("Model-executor", {}),
        "metric_source": deploy_card.get("metric_source", {}),
    }

    # Ensure pp_mb is present in parallelism
    par = card["Model-executor"].get("model_plan_parallelization", {})
    if "pp_mb" not in par:
        par["pp_mb"] = 1
    if "ep" not in par:
        par["ep"] = 1

    return card


# Models from deployment/
MODELS = ["llama-3.1-8b", "llama-3.1-70b", "mixtral-8x7b", "deepseek-v3"]
ARCHS = ["gpu-a100", "tpu-v6"]
PHASES = ["training", "inference"]

# Remove old benchmark cards
print("=== Removing old benchmark cards ===")
for phase in PHASES:
    phase_dir = BENCH / phase
    if phase_dir.exists():
        for f in phase_dir.glob("*.yaml"):
            f.unlink()
            print(f"  REMOVED: {f.relative_to(REPO)}")

# Create new benchmark cards
print("\n=== Creating new benchmark cards ===")
for model in MODELS:
    model_card = load(TC / model / f"{model}.yaml")

    for arch in ARCHS:
        for phase in PHASES:
            deploy_path = DEPLOY / arch / f"{model}-{phase}.yaml"
            deploy_card = load(deploy_path)

            card = merge(model_card, deploy_card, model)

            tag = hw_tag(deploy_card)
            par = deploy_card.get("Model-executor", {}).get("model_plan_parallelization", {})
            ptag = parallelism_tag(par)
            filename = f"{model}-{tag}-{ptag}.yaml"

            out_path = BENCH / phase / filename
            write(out_path, card)

print("\nDone.")
