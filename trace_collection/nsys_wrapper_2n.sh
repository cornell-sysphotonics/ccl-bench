#!/bin/bash

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export GLOBAL_RANK=$SLURM_PROCID

export CUDA_VISIBLE_DEVICES=$LOCAL_RANK

echo "[Wrapper] Global Rank $RANK (Local $LOCAL_RANK) - Device Selection: Using physical GPU $LOCAL_RANK (Logical cuda:0)"

if [ "$LOCAL_RANK" -eq 0 ]; then
    echo "[Wrapper] Rank $RANK Profiling..."
    nsys profile \
        --trace=cuda,nvtx \
        --output=trace_2nodes_rank_${RANK} \
        --force-overwrite=true \
        --stats=true \
        --duration=600 \
        python "$@"
else
    python "$@"
fi