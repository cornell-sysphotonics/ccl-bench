#!/bin/bash
# Topology sweep with a 32-GPU scale-up domain.
#
# Sweeps both intra-node scale-up topology and inter-node scale-out topology
# while holding bandwidths fixed:
#   scale-up:  450 GB/s, 50 ns
#   scale-out: 50 GB/s, 50000 ns

set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep32-dp8-pp4-tp4-perlmutter

GPUS_PER_NODE=32
INTRA_BW=450
INTRA_LAT=50
INTER_BW=50
INTER_LAT=50000
COLLECTIVE_ALGO=halving_doubling

INTRA_TOPOLOGIES=(Switch Ring FullyConnected)
INTER_TOPOLOGIES=(Switch Ring FullyConnected)

echo "=== Topology Sweep: scale-up domain size ${GPUS_PER_NODE} ==="
echo "Trace: $TRACE"
echo "Scale-up BW: ${INTRA_BW} GB/s, Latency: ${INTRA_LAT} ns"
echo "Scale-out BW: ${INTER_BW} GB/s, Latency: ${INTER_LAT} ns"
echo "Collective algo: ${COLLECTIVE_ALGO}"
echo

PREV_OUTDIR="bw_sweep/ccl_bench_sim_bw5_ep32_900_32"
for INTRATOPO in "${INTRA_TOPOLOGIES[@]}"; do
    for INTERTOPO in "${INTER_TOPOLOGIES[@]}"; do
        OUTDIR="$REPO/simulation/examples/topo_sweep_su32_intra/${INTRATOPO}_inter${INTERTOPO}_halving_doubling"
        LOG="$OUTDIR/sweep_stdout.log"
        REUSE_ARGS=()
        if [ -n "$PREV_OUTDIR" ]; then
            REUSE_ARGS=(--reuse-et-from "$PREV_OUTDIR")
        fi

        echo "--- intra=${INTRATOPO}, inter=${INTERTOPO} ---"
        mkdir -p "$OUTDIR"
        python "$REPO/simulation/pipeline.py" \
            --mode comm-only \
            --trace-dir "$TRACE" \
            --output-dir "$OUTDIR" \
            "${REUSE_ARGS[@]}" \
            --gpus-per-node "$GPUS_PER_NODE" \
            --intra-topology "$INTRATOPO" \
            --intra-bandwidth "$INTRA_BW" \
            --intra-latency "$INTRA_LAT" \
            --topology "$INTERTOPO" \
            --bandwidth "$INTER_BW" \
            --latency "$INTER_LAT" \
            --collective-algo "$COLLECTIVE_ALGO" \
            --compute-model gap \
            2>&1 | tee "$LOG"
        grep -E "Simulated step|[Cc]omm fraction|ERROR|Reused .*Chakra ET|Generating Chakra ET" "$LOG" || true
        PREV_OUTDIR="$OUTDIR"
        echo
    done
done
