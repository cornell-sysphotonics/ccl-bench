#!/bin/bash
set -e

echo "[pipeline] Extracting NCCL collectives and generating Chakra ET files..."
python3 /mnt/scripts/gen_chakra_et.py \
    --trace-dir /mnt/traces \
    --output-dir /mnt/output \
    --ranks 0,1,2,3,4,5,6,7 \
    --compute-model kernels

echo "[pipeline] Running AstraSim..."
COMM_GROUP_ARG=""
if [ -f /mnt/output/comm_group.json ]; then
    COMM_GROUP_ARG="--comm-group-configuration=/mnt/output/comm_group.json"
    echo "[pipeline] Using comm_group.json for hybrid-parallelism-aware simulation"
fi

/app/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
    --workload-configuration=/mnt/output/chakra_trace \
    --system-configuration=/mnt/output/system.json \
    --remote-memory-configuration=/mnt/output/remote_memory.json \
    --network-configuration=/mnt/output/network.yml \
    $COMM_GROUP_ARG

echo "[pipeline] Done."
