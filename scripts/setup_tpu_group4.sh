#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <ZONE> <TPU_NAME> <DISK_NAME>" >&2
  exit 1
fi


ZONE=$1
TPU_NAME=$2
DISK_NAME=$3
DISK_SIZE=${DISK_SIZE:-300GB}
MOUNT_PATH=${MOUNT_PATH:-/mnt/disks/huggingface}
HF_HOME=${HF_HOME:-$MOUNT_PATH/hf_home}
DOCKER_URI=vllm/vllm-tpu:v0.12.0
HF_TOKEN=${HF_TOKEN:?set HF_TOKEN for model downloads}
CONTAINER_NAME=${CONTAINER_NAME:-vllm-profile}

MODEL_ID=${MODEL_ID:-meta-llama/Llama-3.3-70B-Instruct}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
TP_SIZE=${TP_SIZE:-8}
PROFILE_DIR=${PROFILE_DIR:-/tmp/tpu_profile/llama}
HOST_PROFILE_DIR=${HOST_PROFILE_DIR:-/tmp/tpu_profile_host/llama}

echo "Ensuring disk $DISK_NAME exists in zone $ZONE..."
if ! gcloud compute disks describe "$DISK_NAME" --zone="$ZONE" >/dev/null 2>&1; then
  gcloud compute disks create "$DISK_NAME" \
    --type=hyperdisk-balanced \
    --size="$DISK_SIZE" \
    --zone="$ZONE" \
    --quiet
fi

echo "Ensuring disk $DISK_NAME is attached to TPU $TPU_NAME..."
if ! gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" \
  --format="value(attachedDisks.disk.name)" | grep -qx "$DISK_NAME"; then
  gcloud alpha compute tpus tpu-vm attach-disk "$TPU_NAME" \
    --zone="$ZONE" \
    --disk="$DISK_NAME" \
    --quiet
fi

REMOTE_SETUP=$(cat <<'EOF'
set -euo pipefail

DISK_DEVICE=$(sudo lsblk -nr -o NAME,TYPE | awk '$2=="disk"{last=$1} END{print last}')
echo "Selected disk device: $DISK_DEVICE"

if ! sudo blkid /dev/${DISK_DEVICE} >/dev/null 2>&1; then
  sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/${DISK_DEVICE}
fi

sudo mkdir -p __MOUNT_PATH__
echo "Created mount point: __MOUNT_PATH__"
sudo mount -o discard,defaults /dev/${DISK_DEVICE} __MOUNT_PATH__

sudo chmod a+w __MOUNT_PATH__

export HF_TOKEN="__HF_TOKEN__"
export HF_HOME="__HF_HOME__"

echo "HF_TOKEN: $HF_TOKEN"

if ! sudo docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon is not accessible on TPU VM. Start it and rerun." >&2
  exit 1
fi

echo "Ensuring container __CONTAINER_NAME__ is running..."
if sudo docker ps -a --format '{{.Names}}' | grep -q "^__CONTAINER_NAME__$"; then
  if ! sudo docker ps --format '{{.Names}}' | grep -q "^__CONTAINER_NAME__$"; then
    sudo docker start "__CONTAINER_NAME__"
  fi
else
  sudo docker run -d --name "__CONTAINER_NAME__" --privileged --net=host \
    -v "__MOUNT_PATH__:__MOUNT_PATH__" \
    -v /dev/shm:/dev/shm \
    --shm-size 250g \
    -e HF_TOKEN="$HF_TOKEN" \
    -e HF_HOME="$HF_HOME" \
    --entrypoint /bin/bash "__DOCKER_URI__" -lc "mkdir -p \"\${HF_HOME}\"; tail -f /dev/null"
fi
EOF
)

REMOTE_SETUP=${REMOTE_SETUP//__MOUNT_PATH__/$MOUNT_PATH}
REMOTE_SETUP=${REMOTE_SETUP//__HF_TOKEN__/$HF_TOKEN}
REMOTE_SETUP=${REMOTE_SETUP//__HF_HOME__/$HF_HOME}
REMOTE_SETUP=${REMOTE_SETUP//__DOCKER_URI__/$DOCKER_URI}
REMOTE_SETUP=${REMOTE_SETUP//__CONTAINER_NAME__/$CONTAINER_NAME}

echo "Running one-time setup (mount + container ensure) on TPU $TPU_NAME..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command "cat <<'REMOTE' | bash
$REMOTE_SETUP
REMOTE"
