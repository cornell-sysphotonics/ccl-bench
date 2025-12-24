# Trace Collection for Qwen/Qwen3-4B (TP=4)

This folder contains scripts to collect traces for the CS5470 project. We simplify the process of setting up the TPU and running the profiling.

## Step 1: Spin up a Google Cloud TPU
Spin up a TPU v6e-8 (required for this trace configuration):
```bash
gcloud alpha compute tpus queued-resources create my-tpu-v6e-queue \
  --node-id my-tpu-v6e \
  --zone=us-east1-d \
  --accelerator-type=v6e-8 \
  --runtime-version=v2-alpha-tpuv6e \
  --network=global-project-net \
  --subnetwork=global-project-net \
  --provisioning-model=SPOT
```

## Step 2: Allocate external storage
Create a disk for storing model weights (needed for large models):
```bash
gcloud compute disks create hf-disk --type=hyperdisk-balanced --size=300GB --zone=us-east1-d
```

## Step 3: Run the TPU setup script
This script mounts the external storage and sets up the Docker container for profiling. **Run this once per TPU.**
```bash
../../scripts/setup_tpu_group-4.sh <ZONE> <TPU_NAME> <DISK_NAME>
# Example:
# ../../scripts/setup_tpu_group-4.sh us-east1-d my-tpu-v6e hf-disk
```

## Step 4: Run the trace collection script
This script runs locally, connects to the TPU via SSH, executes the profiling for various batch sizes, and copies the results back to this folder.
```bash
export HF_TOKEN=your_token
./run.sh <ZONE> <TPU_NAME>
# Example:
# ./run.sh us-east1-d my-tpu-v6e
```
