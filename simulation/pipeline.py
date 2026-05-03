#!/usr/bin/env python3
"""
What-if hardware simulation pipeline for CCL-Bench traces via Astra-sim.

Two modes:
  full      - Requires pytorch_et_N.json + kineto_trace_N.json per rank (full fidelity).
              Available in: trace_collection_backlog/llama-3.1-8b-torchtitan-perlmutter/
  comm-only - Requires rankN_trace.json only (kineto Chrome JSON).
              Works with any trace in /data/ccl-bench_trace_collection/.
              Extracts NCCL collectives + measured compute gaps → Chakra ET → AstraSim.
              When NCCL events carry "Process Group Ranks" (torchtitan traces), the
              pipeline automatically generates comm_group.json and passes it to
              AstraSim, enabling hybrid-parallelism-aware simulation.

Usage:
    python simulation/pipeline.py --trace-dir <path> [--mode comm-only] [hardware overrides]
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DOCKER_IMAGE = "astra-sim:latest"
DOCKER_BIN = "/app/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
SIMULATION_DIR = Path(__file__).parent


def detect_ranks_full(trace_dir: Path) -> list[int]:
    """Detect ranks from pytorch_et_N.json files (full mode)."""
    ranks = []
    for f in sorted(trace_dir.glob("pytorch_et_*.json")):
        m = re.search(r"pytorch_et_(\d+)\.json", f.name)
        if m:
            ranks.append(int(m.group(1)))
    return sorted(ranks)


def detect_ranks_comm_only(trace_dir: Path) -> list[int]:
    """Detect ranks from rankN_trace.json files (comm-only mode)."""
    ranks = []
    for f in sorted(trace_dir.glob("rank*_trace.json")):
        m = re.search(r"rank(\d+)_trace\.json", f.name)
        if m:
            ranks.append(int(m.group(1)))
    return sorted(ranks)


def validate_full_mode(trace_dir: Path, ranks: list[int]):
    missing = []
    for r in ranks:
        for name in [f"pytorch_et_{r}.json", f"kineto_trace_{r}.json"]:
            if not (trace_dir / name).exists():
                missing.append(name)
    if missing:
        print("ERROR: Missing trace files:", ", ".join(missing[:6]))
        print()
        print("Full mode requires pytorch_et_N.json + kineto_trace_N.json per rank.")
        print("Use --mode comm-only for traces that only have rankN_trace.json,")
        print("or use: trace_collection_backlog/llama-3.1-8b-torchtitan-perlmutter/")
        sys.exit(1)


def write_network_config(path: Path, topology: str, npus_count: int,
                         bandwidth_gbps: float, latency_ns: float,
                         gpus_per_node: int, intra_topology: str,
                         intra_bandwidth_gbps: float,
                         intra_latency_ns: float) -> dict[str, list]:
    if gpus_per_node <= 0:
        raise ValueError("--gpus-per-node must be positive")
    if npus_count <= gpus_per_node:
        topologies = [intra_topology]
        npus_counts = [npus_count]
        bandwidths = [intra_bandwidth_gbps]
        latencies = [intra_latency_ns]
    else:
        if npus_count % gpus_per_node != 0:
            raise ValueError(
                f"rank count ({npus_count}) must be divisible by "
                f"--gpus-per-node ({gpus_per_node}) for two-tier topology"
            )
        topologies = [intra_topology, topology]
        npus_counts = [gpus_per_node, npus_count // gpus_per_node]
        bandwidths = [intra_bandwidth_gbps, bandwidth_gbps]
        latencies = [intra_latency_ns, latency_ns]

    def _fmt(values):
        return ", ".join(str(v) for v in values)

    def _fmt_float(values):
        return ", ".join(f"{v:.2f}" for v in values)

    path.write_text(
        f"# Network configuration for Astra-sim what-if analysis\n"
        f"topology: [ {_fmt(topologies)} ]\n"
        f"npus_count: [ {_fmt(npus_counts)} ]\n"
        f"bandwidth: [ {_fmt_float(bandwidths)} ]  # GB/s\n"
        f"latency: [ {_fmt_float(latencies)} ]  # ns\n"
    )
    return {
        "topologies": topologies,
        "npus_counts": npus_counts,
        "bandwidths": bandwidths,
        "latencies": latencies,
    }


def write_system_config(path: Path, collective_algo: str, peak_perf_tflops: float,
                        mem_bw_gbps: float, scheduling_policy: str,
                        active_chunks: int, dataset_splits: int,
                        roofline_enabled: int = 1):
    astra_collective_algo = {
        "halving_doubling": "halvingDoubling",
        "direct_point2point": "direct",
    }.get(collective_algo, collective_algo)
    config = {
        "scheduling-policy": scheduling_policy,
        "endpoint-delay": 10,
        "active-chunks-per-dimension": active_chunks,
        "preferred-dataset-splits": dataset_splits,
        "all-reduce-implementation": [astra_collective_algo],
        "all-gather-implementation": [astra_collective_algo],
        "reduce-scatter-implementation": [astra_collective_algo],
        "all-to-all-implementation": [astra_collective_algo],
        "collective-optimization": "localBWAware",
        "local-mem-bw": mem_bw_gbps,
        "boost-mode": 0,
        "roofline-enabled": roofline_enabled,
        "peak-perf": peak_perf_tflops,
    }
    path.write_text(json.dumps(config, indent=2))


def write_remote_memory_config(path: Path):
    path.write_text(json.dumps({"memory-type": "NO_MEMORY_EXPANSION"}, indent=2))


def build_full_docker_script(trace_dir_docker: str, output_dir_docker: str,
                             ranks: list[int],
                             generate_et: bool = True) -> str:
    link_cmds = "\n".join(
        f"  chakra_trace_link"
        f" --chakra-host-trace {trace_dir_docker}/pytorch_et_{r}.json"
        f" --chakra-device-trace {trace_dir_docker}/kineto_trace_{r}.json"
        f" --rank {r}"
        f" --output-file {output_dir_docker}/linked_{r}.json"
        for r in ranks
    )
    convert_cmds = "\n".join(
        f"  chakra_converter PyTorch"
        f" --input {output_dir_docker}/linked_{r}.json"
        f" --output {output_dir_docker}/chakra_trace.{r}"
        for r in ranks
    )
    if generate_et:
        et_step = f"""echo "[pipeline] Linking host + device traces..."
{link_cmds}

echo "[pipeline] Converting to Chakra ET format..."
{convert_cmds}
"""
    else:
        et_step = 'echo "[pipeline] Reusing existing Chakra ET files..."\n'

    return f"""#!/bin/bash
set -e

{et_step}

echo "[pipeline] Running AstraSim..."
{DOCKER_BIN} \\
    --workload-configuration={output_dir_docker}/chakra_trace \\
    --system-configuration={output_dir_docker}/system.json \\
    --remote-memory-configuration={output_dir_docker}/remote_memory.json \\
    --network-configuration={output_dir_docker}/network.yml

echo "[pipeline] Done."
"""


def build_comm_only_docker_script(trace_dir_docker: str, output_dir_docker: str,
                                  scripts_dir_docker: str, ranks: list[int],
                                  compute_model: str,
                                  kernel_dependency_mode: str,
                                  generate_et: bool = True) -> str:
    ranks_str = ",".join(str(r) for r in ranks)
    # comm_group.json is written by gen_chakra_et.py when process group info is
    # present in the traces; otherwise the file is absent and we skip the flag.
    if generate_et:
        et_step = f"""echo "[pipeline] Extracting NCCL collectives and generating Chakra ET files..."
python3 {scripts_dir_docker}/gen_chakra_et.py \\
    --trace-dir {trace_dir_docker} \\
    --output-dir {output_dir_docker} \\
    --ranks {ranks_str} \\
    --compute-model {compute_model} \\
    --kernel-dependency-mode {kernel_dependency_mode}
"""
    else:
        et_step = 'echo "[pipeline] Reusing existing Chakra ET files..."\n'

    return f"""#!/bin/bash
set -e

{et_step}

echo "[pipeline] Running AstraSim..."
COMM_GROUP_ARG=""
if [ -f {output_dir_docker}/comm_group.json ]; then
    COMM_GROUP_ARG="--comm-group-configuration={output_dir_docker}/comm_group.json"
    echo "[pipeline] Using comm_group.json for hybrid-parallelism-aware simulation"
fi

{DOCKER_BIN} \\
    --workload-configuration={output_dir_docker}/chakra_trace \\
    --system-configuration={output_dir_docker}/system.json \\
    --remote-memory-configuration={output_dir_docker}/remote_memory.json \\
    --network-configuration={output_dir_docker}/network.yml \\
    $COMM_GROUP_ARG

echo "[pipeline] Done."
"""


def run_in_docker(trace_dir: Path, output_dir: Path, script: str,
                  extra_mounts: list[tuple[Path, str]] | None = None) -> int:
    script_path = output_dir / "run_pipeline.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)

    cmd = [
        "docker", "run", "--rm",
        # Run from /mnt/output so AstraSim writes any files there
        "-w", "/mnt/output",
        "-v", f"{trace_dir.resolve()}:/mnt/traces:ro",
        "-v", f"{output_dir.resolve()}:/mnt/output",
    ]
    for host_path, docker_path in (extra_mounts or []):
        cmd += ["-v", f"{host_path.resolve()}:{docker_path}:ro"]
    cmd += [DOCKER_IMAGE, "/bin/bash", "/mnt/output/run_pipeline.sh"]

    log_path = output_dir / "simulation.log"
    print(f"[pipeline] docker run ... (log: {log_path})")
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        log_f.write(result.stdout)
        print(result.stdout, end="")
    return result.returncode


def _parse_astrasim_stats(log_path: Path) -> dict[int, dict[str, int]]:
    """Extract per-sys statistics from AstraSim log output."""
    stats: dict[int, dict[str, int]] = {}
    for line in log_path.read_text().splitlines():
        if "[statistics]" not in line:
            continue
        # e.g. "sys[6], Wall time: 8559736399"
        import re
        m = re.search(r"sys\[(\d+)\], (\w[\w ]+): (\d+)", line)
        if m:
            sys_id = int(m.group(1))
            key = m.group(2).strip()
            val = int(m.group(3))
            stats.setdefault(sys_id, {})[key] = val
    return stats


def show_results(output_dir: Path):
    print("\n[pipeline] Output files:")
    for f in sorted(output_dir.iterdir()):
        if f.suffix in (".et", ".log", ".json", ".yml") and f.stat().st_size > 0:
            print(f"  {f.name} ({f.stat().st_size:,} bytes)")

    log_path = output_dir / "simulation.log"
    if not log_path.exists():
        return

    stats = _parse_astrasim_stats(log_path)
    if not stats:
        return

    print("\n[pipeline] AstraSim simulation results (ns):")
    header = f"  {'sys':>4}  {'Wall time':>16}  {'Comm time':>16}  {'GPU time':>16}  {'Comm %':>7}"
    print(header)
    for sys_id in sorted(stats):
        s = stats[sys_id]
        wall = s.get("Wall time", 0)
        comm = s.get("Comm time", 0)
        gpu = s.get("GPU time", 0)
        comm_pct = 100.0 * comm / wall if wall > 0 else 0
        print(f"  {sys_id:>4}  {wall:>16,}  {comm:>16,}  {gpu:>16,}  {comm_pct:>6.1f}%")

    if stats:
        walls = [s.get("Wall time", 0) for s in stats.values()]
        comms = [s.get("Comm time", 0) for s in stats.values()]
        avg_wall = sum(walls) / len(walls)
        avg_comm = sum(comms) / len(comms)
        avg_comm_pct = 100.0 * avg_comm / avg_wall if avg_wall > 0 else 0
        print(f"  {'avg':>4}  {avg_wall:>16,.0f}  {avg_comm:>16,.0f}  {'':>16}  {avg_comm_pct:>6.1f}%")
        print(f"\n  Simulated step time: {avg_wall/1e6:.1f} ms  |  Comm fraction: {avg_comm_pct:.1f}%")


def copy_et_artifacts(src_dir: Path, dst_dir: Path) -> None:
    """Copy reusable Chakra ET workload artifacts into a new output directory."""
    et_files = sorted(src_dir.glob("chakra_trace.*.et"))
    if not et_files:
        print(f"ERROR: no chakra_trace.*.et files found in reuse source: {src_dir}")
        sys.exit(1)

    for path in et_files:
        shutil.copy2(path, dst_dir / path.name)

    comm_group = src_dir / "comm_group.json"
    dst_comm_group = dst_dir / "comm_group.json"
    if comm_group.exists():
        shutil.copy2(comm_group, dst_comm_group)
    elif dst_comm_group.exists():
        dst_comm_group.unlink()

    print(f"[pipeline] Reused {len(et_files)} Chakra ET file(s) from {src_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="What-if hardware simulation for CCL-Bench traces via Astra-sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full       Requires pytorch_et_N.json + kineto_trace_N.json (full op graph fidelity)
             → trace_collection_backlog/llama-3.1-8b-torchtitan-perlmutter/
  comm-only  Requires rankN_trace.json only (any torchtitan trace in /data/)
             Extracts NCCL collectives and either compute gaps or non-NCCL
             kernel replay nodes. Compute time is replayed as measured; only
             communication time varies with hardware params.
             Torchtitan traces (deepseek-v3, etc.) include process group info and
             automatically enable hybrid-parallelism-aware simulation via comm_group.json.
             → /data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-*/

Examples:
  # Full mode, baseline A100 Slingshot
  python simulation/pipeline.py --mode full \\
      --trace-dir trace_collection_backlog/llama-3.1-8b-torchtitan-perlmutter

  # Comm-only: deepseek trace, what-if 2× bandwidth
  python simulation/pipeline.py --mode comm-only \\
      --trace-dir /data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep4-dp2-pp2-tp4-perlmutter \\
      --bandwidth 400

  # Comm-only: replay actual non-NCCL GPU kernels as compute nodes
  python simulation/pipeline.py --mode comm-only \\
      --trace-dir /data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep4-dp2-pp2-tp4-perlmutter \\
      --compute-model kernels

  # Comm-only: compare ring vs halving_doubling collective algorithm
  python simulation/pipeline.py --mode comm-only \\
      --trace-dir /data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep4-dp2-pp2-tp4-perlmutter \\
      --collective-algo halving_doubling --bandwidth 200

  # Full mode, what-if H100 hardware (900 TFLOPS BF16, 3.35 TB/s HBM)
  python simulation/pipeline.py --mode full \\
      --trace-dir trace_collection_backlog/llama-3.1-8b-torchtitan-perlmutter \\
      --peak-perf 989 --mem-bw 3350 --gpus-per-node 8 \\
      --intra-bandwidth 900 --bandwidth 50
""",
    )
    parser.add_argument("--trace-dir", required=True,
                        help="Trace directory path")
    parser.add_argument("--mode", default="full", choices=["full", "comm-only"],
                        help="full: needs pytorch_et+kineto; comm-only: needs rankN_trace.json only")
    parser.add_argument("--compute-model", default="gap", choices=["gap", "kernels"],
                        help="comm-only compute model: gap inserts measured gaps between "
                             "collectives; kernels replays non-NCCL GPU kernels as "
                             "COMP_NODEs (default: gap)")
    parser.add_argument("--kernel-dependency-mode", default="rank",
                        choices=["rank", "stream"],
                        help="comm-only kernels model dependency mode: rank adds "
                             "rank-local chronological dependencies between compute "
                             "and communication kernels; stream preserves only CUDA "
                             "stream ordering (default: rank)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: /var/tmp/ccl_bench_sim_*)")
    parser.add_argument("--reuse-et-from", default=None,
                        help="reuse chakra_trace.*.et and optional comm_group.json "
                             "from this output directory instead of regenerating ET")

    net = parser.add_argument_group("Network what-if parameters")
    net.add_argument("--topology", default="Switch",
                     choices=["Ring", "Switch", "FullyConnected"],
                     help="Inter-node network topology (default: Switch)")
    net.add_argument("--bandwidth", type=float, default=200.0,
                     help="Inter-node per-link bandwidth GB/s (default: 200, A100 Slingshot)")
    net.add_argument("--latency", type=float, default=500.0,
                     help="Inter-node per-link latency ns (default: 500)")
    net.add_argument("--gpus-per-node", type=int, default=4,
                     help="Ranks/GPUs that share the intra-node fabric (default: 4, Perlmutter A100)")
    net.add_argument("--intra-topology", default="FullyConnected",
                     choices=["Ring", "Switch", "FullyConnected"],
                     help="Intra-node topology (default: FullyConnected)")
    net.add_argument("--intra-bandwidth", type=float, default=400.0,
                     help="Intra-node per-link bandwidth GB/s (default: 400, NVLink-class)")
    net.add_argument("--intra-latency", type=float, default=50.0,
                     help="Intra-node per-link latency ns (default: 50)")

    sys_grp = parser.add_argument_group("System what-if parameters")
    sys_grp.add_argument("--collective-algo", default="ring",
                         choices=["ring", "halving_doubling", "direct_point2point",
                                  "doubleBinaryTree"],
                         help="Collective algorithm (default: ring)")
    sys_grp.add_argument("--peak-perf", type=float, default=312.0,
                         help="Peak compute TFLOPS (default: 312, A100 BF16)")
    sys_grp.add_argument("--mem-bw", type=float, default=2000.0,
                         help="HBM bandwidth GB/s (default: 2000, A100)")
    sys_grp.add_argument("--scheduling-policy", default="LIFO",
                         choices=["LIFO", "FIFO"])
    sys_grp.add_argument("--active-chunks", type=int, default=1)
    sys_grp.add_argument("--dataset-splits", type=int, default=4)

    args = parser.parse_args()

    trace_dir = Path(args.trace_dir).resolve()
    if not trace_dir.exists():
        print(f"ERROR: trace directory not found: {trace_dir}")
        sys.exit(1)

    # Detect ranks and validate inputs
    if args.mode == "full":
        ranks = detect_ranks_full(trace_dir)
        if not ranks:
            print(f"ERROR: No pytorch_et_N.json found in {trace_dir}")
            print("Use --mode comm-only for traces with only rankN_trace.json")
            sys.exit(1)
        validate_full_mode(trace_dir, ranks)
    else:
        ranks = detect_ranks_comm_only(trace_dir)
        if not ranks:
            print(f"ERROR: No rankN_trace.json files found in {trace_dir}")
            sys.exit(1)
        print(f"[pipeline] comm-only mode: compute gaps replayed as measured; "
              f"only communication time varies with hardware parameters")

    print(f"[pipeline] Mode: {args.mode}, {len(ranks)} ranks")

    # Set up output directory under /var/tmp to avoid /tmp pressure
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="ccl_bench_sim_", dir="/var/tmp"))
        print(f"[pipeline] Tip: use --output-dir to keep outputs between runs")

    print(f"[pipeline] Output: {output_dir}")

    reuse_et_from = Path(args.reuse_et_from).resolve() if args.reuse_et_from else None
    if reuse_et_from is not None:
        if not reuse_et_from.exists():
            print(f"ERROR: --reuse-et-from directory not found: {reuse_et_from}")
            sys.exit(1)
        copy_et_artifacts(reuse_et_from, output_dir)

    # Write hardware configs
    # comm-only disables roofline for compute (compute time comes from duration_micros)
    roofline = 0 if args.mode == "comm-only" else 1
    try:
        network_shape = write_network_config(
            output_dir / "network.yml", args.topology, len(ranks),
            args.bandwidth, args.latency, args.gpus_per_node,
            args.intra_topology, args.intra_bandwidth, args.intra_latency
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    write_system_config(output_dir / "system.json", args.collective_algo,
                        args.peak_perf, args.mem_bw, args.scheduling_policy,
                        args.active_chunks, args.dataset_splits,
                        roofline_enabled=roofline)
    write_remote_memory_config(output_dir / "remote_memory.json")

    if len(network_shape["npus_counts"]) == 1:
        print(f"[pipeline] Hardware: topology={network_shape['topologies'][0]}, "
              f"npus={len(ranks)}, "
              f"bw={network_shape['bandwidths'][0]}GB/s, "
              f"latency={network_shape['latencies'][0]}ns, "
              f"algo={args.collective_algo}")
    else:
        print(f"[pipeline] Hardware: topology={network_shape['topologies']}, "
              f"npus={network_shape['npus_counts'][0]}x{network_shape['npus_counts'][1]} "
              f"(gpus/node={args.gpus_per_node}), "
              f"bw={network_shape['bandwidths']}GB/s, "
              f"latency={network_shape['latencies']}ns, "
              f"algo={args.collective_algo}")

    # Build Docker run
    if args.mode == "full":
        script = build_full_docker_script(
            "/mnt/traces", "/mnt/output", ranks,
            generate_et=(reuse_et_from is None)
        )
        extra_mounts = None
    else:
        script = build_comm_only_docker_script(
            "/mnt/traces", "/mnt/output", "/mnt/scripts", ranks,
            args.compute_model, args.kernel_dependency_mode,
            generate_et=(reuse_et_from is None)
        )
        extra_mounts = [(SIMULATION_DIR, "/mnt/scripts")]

    ret = run_in_docker(trace_dir, output_dir, script, extra_mounts=extra_mounts)
    if ret != 0:
        print(f"[pipeline] ERROR: Docker run failed (exit {ret})")
        sys.exit(ret)

    show_results(output_dir)
    print("\n[pipeline] Complete.")


if __name__ == "__main__":
    main()
