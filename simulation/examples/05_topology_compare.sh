#!/bin/bash
# Topology comparison: Switch (fat-tree) vs Ring vs FullyConnected.
#
# Compares the inter-node scale-out topology while keeping intra-node NVLink-like
# scale-up fixed.
#
# Switch: every node connects to a central switch. Bandwidth is per-link
#   (NPU→switch), so effective all-reduce bandwidth ≈ BW/2 for large messages.
# Ring: nodes form a ring. Optimal for bandwidth-bound, latency-sensitive for small messages.
# FullyConnected: direct node-to-node links; models high-radix scale-out networks.
#
# Same scale-out bandwidth budget (200 GB/s per link) across all topologies.
# Trace: deepseek-v3-16b ep8-dp8 (8 ranks, EP-heavy, no TP or PP).

set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep8-dp8-perlmutter
GPUS_PER_NODE=4
INTRA_BW=400
INTRA_LAT=50
INTER_BW=200
INTER_LAT=500

echo "=== Topology Comparison: deepseek-v3-16b ep8-dp8 (8 GPUs) ==="
echo "Fixed scale-up: FullyConnected, ${INTRA_BW} GB/s, ${INTRA_LAT} ns"
echo "Scale-out BW: ${INTER_BW} GB/s per link, Latency: ${INTER_LAT} ns, algo: ring"
echo

for TOPO in Switch Ring FullyConnected; do
    OUTDIR=/var/tmp/ccl_bench_sim_topo_${TOPO}

    echo "--- $TOPO ---"
    python "$REPO/simulation/pipeline.py" \
        --mode comm-only \
        --trace-dir "$TRACE" \
        --output-dir "$OUTDIR" \
        --gpus-per-node "$GPUS_PER_NODE" \
        --intra-topology FullyConnected \
        --intra-bandwidth "$INTRA_BW" \
        --intra-latency "$INTRA_LAT" \
        --topology "$TOPO" \
        --bandwidth "$INTER_BW" \
        --latency "$INTER_LAT" \
        --collective-algo ring \
        2>&1 | grep -E "Simulated step|comm fraction|ERROR"
    echo
done
