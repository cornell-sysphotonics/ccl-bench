#!/usr/bin/env python3
import argparse
import os
import subprocess
import shutil
from pathlib import Path


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True)


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def build_remote_run(
    hf_token: str,
    hf_home: str,
    container_name: str,
    model_id: str,
    input_len: int,
    output_len: int,
    batch_size: int,
    tp_size: int,
    profile_dir: str,
    host_profile_dir: str,
    gpu_mem_util: float,
    max_model_len: int,
) -> str:
    return f"""set -euo pipefail

export HF_TOKEN="{hf_token}"
export HF_HOME="{hf_home}"

if ! sudo docker ps --format '{{{{.Names}}}}' | grep -q "^{container_name}$"; then
  echo "Error: container {container_name} is not running" >&2
  exit 1
fi

sudo docker exec -e HF_TOKEN="$HF_TOKEN" -e HF_HOME="$HF_HOME" {container_name} /bin/bash -lc "\\
rm -rf {profile_dir} && \\
mkdir -p \\\"$HF_HOME\\\" && \\
echo \\\"Token in container: $HF_TOKEN\\\" && \\
python3 examples/tpu_profiling.py \\
  --model {model_id} \\
  --input-len {input_len} \\
  --output-len {output_len} \\
  --batch-size {batch_size} \\
  --tensor-parallel-size {tp_size} \\
  --max-model-len {max_model_len} \\
  --gpu-memory-utilization {gpu_mem_util} \\
  --profile-result-dir {profile_dir} \\
  --hf-token \\\"$HF_TOKEN\\\" \\
"

HOST_PROFILE_DIR="{host_profile_dir}"
sudo rm -rf "$HOST_PROFILE_DIR"
sudo mkdir -p "$(dirname "$HOST_PROFILE_DIR")"
sudo docker cp "{container_name}:{profile_dir}" "$HOST_PROFILE_DIR"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run profiling for a specific model and TP size.")
    parser.add_argument("--zone", required=True, help="GCP Zone")
    parser.add_argument("--tpu", required=True, help="TPU Name")
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--tp", type=int, required=True, help="Tensor Parallel size")
    args = parser.parse_args()

    # Env/config
    zone = args.zone
    tpu = args.tpu
    model_id = args.model
    tp_size = args.tp
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN is required in the environment")

    # Default constants (can be exposed as args if needed)
    mount_path = os.environ.get("MOUNT_PATH", "/mnt/disks/huggingface")
    hf_home = os.environ.get("HF_HOME", f"{mount_path}/hf_home")
    container_name = os.environ.get("CONTAINER_NAME", "vllm-profile")
    input_len = int(os.environ.get("INPUT_LEN", "1024"))
    output_len = int(os.environ.get("OUTPUT_LEN", "1"))
    profile_dir_base = os.environ.get("PROFILE_DIR", "/tmp/tpu_profile")
    host_profile_dir_base = os.environ.get("HOST_PROFILE_DIR", "/tmp/tpu_profile_host")
    error_log = Path(os.environ.get("ERROR_LOG", "profiling_errors.log"))
    gpu_mem_util = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"))
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "1025"))

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    safe_model_id = model_id.replace("/", "_")

    print(f"Starting profiling for {model_id} with TP={tp_size} on {tpu} ({zone})")

    for batch_size in batch_sizes:
        current_profile_dir = f"{profile_dir_base}/{safe_model_id}_tp{tp_size}_batch{batch_size}"
        current_host_profile_dir = f"{host_profile_dir_base}/{safe_model_id}_tp{tp_size}_batch{batch_size}"
        result_dir_name = f"MODEL_{safe_model_id},INPUT_{input_len},OUTPUT_{output_len},BATCH_{batch_size},TP_{tp_size}"

        if os.path.exists(result_dir_name):
             print(f"[SKIP] Result dir {result_dir_name} already exists in current directory.")
             continue

        print("=" * 42)
        print(f"Running profiling with TP={tp_size}, batch={batch_size}, model={model_id}")
        print("=" * 42)

        remote_script = build_remote_run(
            hf_token=hf_token,
            hf_home=hf_home,
            container_name=container_name,
            model_id=model_id,
            input_len=input_len,
            output_len=output_len,
            batch_size=batch_size,
            tp_size=tp_size,
            profile_dir=current_profile_dir,
            host_profile_dir=current_host_profile_dir,
            gpu_mem_util=gpu_mem_util,
            max_model_len=max_model_len,
        )

        ssh_cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            tpu,
            f"--zone={zone}",
            "--command",
            f"cat <<'REMOTE' | bash\n{remote_script}\nREMOTE",
        ]

        result = run_cmd(ssh_cmd)
        if result.returncode != 0:
            print(f"[SKIP] TP={tp_size}, batch={batch_size} failed (likely OOM/HBM).")
            print("---- stdout ----")
            print(result.stdout)
            print("---- stderr ----")
            print(result.stderr)
            
            append_log(
                error_log,
                f"tp={tp_size},batch={batch_size}: rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}\n---\n",
            )
            # If OOM, we might want to stop larger batches, but let's continue for now as per original script logic (it had a break, but maybe we want to try others?)
            # Original script had: if "RESOURCE_EXHAUSTED" ... break.
            if "RESOURCE_EXHAUSTED" in result.stderr:
                 print("Stopping larger batches due to OOM.")
                 break
            continue

        scp_cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "scp",
            f"{tpu}:{current_host_profile_dir}",
            f"./{result_dir_name}",
            f"--zone={zone}",
            "--recurse",
        ]

        scp_result = run_cmd(scp_cmd)
        if scp_result.returncode != 0:
            print(f"[SKIP COPY] TP={tp_size}, batch={batch_size} copy failed.")
            print("---- stdout ----")
            print(scp_result.stdout)
            print("---- stderr ----")
            print(scp_result.stderr)
            append_log(
                error_log,
                f"tp={tp_size},batch={batch_size} copy failed: rc={scp_result.returncode}\nstdout:\n{scp_result.stdout}\nstderr:\n{scp_result.stderr}\n---\n",
            )
            continue
        
        print(f"Successfully collected trace for batch={batch_size}")

if __name__ == "__main__":
    main()
