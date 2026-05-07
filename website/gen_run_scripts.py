#!/usr/bin/env python3
"""
Generate run.sh for every trace_collection subdirectory that lacks one.
Run from the repository root with: conda run -n opus python3 website/gen_run_scripts.py
"""

import yaml
import stat
import sys
from pathlib import Path

TRACE_COLL = Path("trace_collection")

# ── Template builders ──────────────────────────────────────────────────────────

def vllm_script(name: str, meta: dict) -> str:
    model = meta.get("model_family", "unknown-model")
    hf_id = meta.get("hf_url", "").split("huggingface.co/")[-1] or model
    tp = meta.get("tp") or 1
    ep = meta.get("ep") or 1
    batch = meta.get("batch_size") or 8
    input_len = meta.get("input_len") or 1024
    output_len = meta.get("output_len") or 128
    gpu_count = meta.get("total_count") or tp
    comm_env = meta.get("comm_env") or {}
    env_lines = "\n".join(f'export {k}="{v}"' for k, v in comm_env.items())

    return f"""#!/bin/bash
# Run script for {name}
# Framework: vLLM  |  Model: {model}  |  TP={tp}  EP={ep}
set -euo pipefail

MODEL="{hf_id}"
TP={tp}
EP={ep}
BATCH_SIZE={batch}
INPUT_LEN={input_len}
OUTPUT_LEN={output_len}
NUM_GPUS={gpu_count}
PORT=8000

{env_lines}

# Launch vLLM server with Nsight Systems profiling
nsys profile \\
  --output={name}_nsys \\
  --trace=cuda,nvtx,osrt \\
  --capture-range=cudaProfilerApi \\
  --force-overwrite=true \\
  python -m vllm.entrypoints.openai.api_server \\
    --model "$MODEL" \\
    --tensor-parallel-size $TP \\
    --enable-expert-parallel \\
    --max-num-seqs $BATCH_SIZE \\
    --port $PORT &

SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM server..."
for i in $(seq 1 120); do
  if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server ready."
    break
  fi
  sleep 5
done

# Run benchmark
python -m vllm.entrypoints.benchmark_serving \\
  --model "$MODEL" \\
  --backend vllm \\
  --endpoint /v1/completions \\
  --num-prompts $BATCH_SIZE \\
  --input-len $INPUT_LEN \\
  --output-len $OUTPUT_LEN

kill $SERVER_PID || true
"""


def sglang_script(name: str, meta: dict) -> str:
    model = meta.get("model_family", "unknown-model")
    hf_id = meta.get("hf_url", "").split("huggingface.co/")[-1] or model
    tp = meta.get("tp") or 1
    ep = meta.get("ep") or 1
    batch = meta.get("batch_size") or 8
    input_len = meta.get("input_len") or 1024
    output_len = meta.get("output_len") or 128
    gpu_count = meta.get("total_count") or tp
    comm_env = meta.get("comm_env") or {}
    env_lines = "\n".join(f'export {k}="{v}"' for k, v in comm_env.items())

    return f"""#!/bin/bash
# Run script for {name}
# Framework: SGLang  |  Model: {model}  |  TP={tp}  EP={ep}
set -euo pipefail

MODEL="{hf_id}"
TP={tp}
EP={ep}
BATCH_SIZE={batch}
INPUT_LEN={input_len}
OUTPUT_LEN={output_len}
PORT=30000

{env_lines}

# Launch SGLang server with Nsight Systems profiling
nsys profile \\
  --output={name}_nsys \\
  --trace=cuda,nvtx,osrt \\
  --capture-range=cudaProfilerApi \\
  --force-overwrite=true \\
  python -m sglang.launch_server \\
    --model-path "$MODEL" \\
    --tp {tp} \\
    --ep {ep} \\
    --port $PORT &

SERVER_PID=$!

echo "Waiting for SGLang server..."
for i in $(seq 1 120); do
  if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server ready."
    break
  fi
  sleep 5
done

# Run benchmark
python -m sglang.bench_serving \\
  --backend sglang \\
  --base-url http://localhost:$PORT \\
  --num-prompts $BATCH_SIZE \\
  --input-len $INPUT_LEN \\
  --output-len $OUTPUT_LEN

kill $SERVER_PID || true
"""


def torchtitan_script(name: str, meta: dict, toml_file: str | None) -> str:
    model = meta.get("model_family", "llama3")
    tp = meta.get("tp") or 1
    dp_shard = meta.get("dp_shard") or 1
    dp_rep = meta.get("dp_replicate") or 1
    pp = meta.get("pp") or 1
    ep = meta.get("ep") or 1
    gpu_count = meta.get("total_count") or 1
    gpus_per_node = meta.get("count_per_node") or 4
    num_nodes = max(1, int(gpu_count) // int(gpus_per_node))

    toml_arg = f"--job.config_file trace_collection/{name}/{toml_file}" if toml_file else ""

    return f"""#!/bin/bash
# Run script for {name}
# Framework: TorchTitan  |  Model: {model}  |  TP={tp}  DP_shard={dp_shard}  PP={pp}  EP={ep}
set -euo pipefail

NUM_NODES={num_nodes}
GPUS_PER_NODE={gpus_per_node}
MASTER_ADDR=${{MASTER_ADDR:-localhost}}
MASTER_PORT=${{MASTER_PORT:-6000}}
NODE_RANK=${{NODE_RANK:-0}}

# Activate torchtitan environment and run
torchrun \\
  --nproc_per_node=$GPUS_PER_NODE \\
  --nnodes=$NUM_NODES \\
  --node_rank=$NODE_RANK \\
  --master_addr=$MASTER_ADDR \\
  --master_port=$MASTER_PORT \\
  torchtitan/train.py \\
  {toml_arg} \\
  --parallelism.tensor_parallel_degree {tp} \\
  --parallelism.data_parallel_shard_degree {dp_shard} \\
  --parallelism.data_parallel_replicate_degree {dp_rep} \\
  --parallelism.pipeline_parallel_degree {pp} \\
  --parallelism.expert_parallel_degree {ep} \\
  --profiling.enable_profiling true \\
  --profiling.save_traces_folder ./profile_traces
"""


def maxtext_script(name: str, meta: dict) -> str:
    model = meta.get("model_family", "deepseek-v2")
    tp = meta.get("tp") or 1
    ep = meta.get("ep") or 1
    dp_shard = meta.get("dp_shard") or 1
    batch = meta.get("batch_size") or 1
    seq_len = meta.get("seq_len") or 1024
    total = meta.get("total_count") or 8

    return f"""#!/bin/bash
# Run script for {name}
# Framework: MaxText  |  Model: {model}  |  TP={tp}  EP={ep}
set -euo pipefail

MAXTEXT_ROOT="${{MAXTEXT_ROOT:-./MaxText}}"
MODEL_NAME="{model}"
TP={tp}
EP={ep}
PER_DEVICE_BATCH={batch}
SEQ_LEN={seq_len}
NUM_DEVICES={total}

python $MAXTEXT_ROOT/train.py \\
  MaxText/configs/base.yml \\
  model_name=$MODEL_NAME \\
  ici_mesh_shape="[1, $TP, 1, $EP]" \\
  per_device_batch_size=$PER_DEVICE_BATCH \\
  max_target_length=$SEQ_LEN \\
  steps=10 \\
  enable_profiler=true \\
  profile_dir=./profiles
"""


def torchxla_train_script(name: str, meta: dict) -> str:
    """For torchxla training workloads (group_21 etc.)"""
    model = meta.get("model_family", "llama-3.1-8b")
    tp = meta.get("tp") or 1
    dp_shard = meta.get("dp_shard") or 1
    batch = meta.get("batch_size") or 4
    seq_len = meta.get("seq_len") or 512
    tpu_model = meta.get("hardware_model") or "tpu_v6e"
    total = meta.get("total_count") or 4

    return f"""#!/bin/bash
# Run script for {name}
# Framework: TorchXLA  |  Model: {model}  |  TP={tp}  FSDP={dp_shard}
# Usage: $0 <ZONE> <TPU_NAME>
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <ZONE> <TPU_NAME>" >&2
  exit 1
fi

ZONE=$1
TPU_NAME=$2
MODEL="{model}"
TP={tp}
FSDP={dp_shard}
BATCH_SIZE={batch}
SEQ_LEN={seq_len}

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \\
  --zone "$ZONE" \\
  --command "
    cd ~/
    python3 train_llm_xla.py \\
      --model $MODEL \\
      --tensor-parallel $TP \\
      --fsdp $FSDP \\
      --batch-size $BATCH_SIZE \\
      --seq-len $SEQ_LEN \\
      --profile
  "
"""


def torchxla_inference_script(name: str, meta: dict) -> str:
    """For torchxla inference workloads (group-4 style, but these already have run.sh)."""
    model = meta.get("model_family", "llama-3.1-8b")
    hf_id = meta.get("hf_url", "").split("huggingface.co/")[-1] or model
    tp = meta.get("tp") or 1

    return f"""#!/bin/bash
# Run script for {name}
# Framework: TorchXLA (vLLM inference)  |  Model: {model}  |  TP={tp}
# Usage: $0 <ZONE> <TPU_NAME>
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <ZONE> <TPU_NAME>" >&2
  exit 1
fi

ZONE=$1
TPU_NAME=$2
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/../.."

python3 $ROOT_DIR/scripts/run_trace-group4.py \\
  --zone "$ZONE" \\
  --tpu "$TPU_NAME" \\
  --model "{hf_id}" \\
  --tp {tp}
"""


def pytorch_script(name: str, meta: dict) -> str:
    gpu_count = meta.get("total_count") or 1
    return f"""#!/bin/bash
# Run script for {name}
# Framework: PyTorch
set -euo pipefail

NUM_GPUS={gpu_count}

torchrun \\
  --nproc_per_node=$NUM_GPUS \\
  train.py
"""


# ── Metadata extraction ────────────────────────────────────────────────────────

def load_meta(yaml_path: Path) -> tuple[str, dict]:
    """Returns (framework_name, flat_meta_dict)."""
    try:
        data = yaml.safe_load(yaml_path.read_text()) or {}
    except Exception as e:
        print(f"  WARN: could not parse {yaml_path.name}: {e}")
        return "unknown", {}

    ex = data.get("Model-executor") or {}
    fw = (ex.get("framework") or {})
    fw_name = (fw.get("name") or "unknown").lower()
    par = ex.get("model_plan_parallelization") or {}
    cl = ex.get("communication_library") or {}
    hw = (data.get("workload") or {}).get("hardware") or {}
    xpu = hw.get("xpu_spec") or {}
    net = hw.get("network_topo") or {}
    mod = (data.get("workload") or {}).get("model") or {}
    dat = (data.get("workload") or {}).get("data") or {}

    meta = {
        "hf_url": data.get("hf_url") or "",
        "model_family": mod.get("model_family") or "",
        "phase": mod.get("phase") or "training",
        "tp": par.get("tp"),
        "pp": par.get("pp"),
        "ep": par.get("ep"),
        "dp_shard": par.get("dp_shard"),
        "dp_replicate": par.get("dp_replicate"),
        "total_count": xpu.get("total_count"),
        "count_per_node": xpu.get("count_per_node"),
        "hardware_model": xpu.get("model"),
        "batch_size": dat.get("batch_size"),
        "seq_len": dat.get("seq_len"),
        "input_len": dat.get("input_len"),
        "output_len": dat.get("output_len"),
        "comm_env": cl.get("env") or {},
    }
    return fw_name, meta


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not TRACE_COLL.is_dir():
        print(f"Error: {TRACE_COLL} not found. Run from repo root.")
        sys.exit(1)

    dirs = sorted(d for d in TRACE_COLL.iterdir() if d.is_dir())
    created = 0
    skipped = 0

    for d in dirs:
        run_sh = d / "run.sh"
        if run_sh.exists():
            skipped += 1
            continue

        yamls = sorted(d.glob("*.yaml"))
        if not yamls:
            print(f"  SKIP {d.name}: no YAML")
            continue

        fw_name, meta = load_meta(yamls[0])
        name = d.name

        # Find .toml config if present (for torchtitan)
        tomls = sorted(d.glob("*.toml"))
        toml_file = tomls[0].name if tomls else None

        if "torchtitan" in fw_name:
            script = torchtitan_script(name, meta, toml_file)
        elif fw_name == "vllm":
            script = vllm_script(name, meta)
        elif fw_name == "sglang":
            script = sglang_script(name, meta)
        elif fw_name == "maxtext":
            script = maxtext_script(name, meta)
        elif fw_name == "torchxla":
            phase = meta.get("phase", "training").lower()
            if phase == "inference":
                script = torchxla_inference_script(name, meta)
            else:
                script = torchxla_train_script(name, meta)
        elif fw_name in ("pytorch", "megatron-lm", "megatron_lm"):
            script = pytorch_script(name, meta)
        else:
            print(f"  SKIP {name}: unknown framework '{fw_name}'")
            continue

        run_sh.write_text(script)
        run_sh.chmod(run_sh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"  + {name}  [{fw_name}]")
        created += 1

    print(f"\nDone: {created} created, {skipped} already existed.")


if __name__ == "__main__":
    main()
