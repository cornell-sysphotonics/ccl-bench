"""
Metric: scale_up_bw_utility
Description: Percent step-time improvement from doubling scale-up (intra-node)
             bandwidth, via Astra-sim simulation. Quantifies how sensitive a
             workload is to the scale-up interconnect bandwidth.

             Runs two simulations via simulation/pipeline.py --mode comm-only:
               1. Baseline: hardware BW read from workload YAML
               2. Variant:  scale-up BW doubled (2× intra-node BW)

             Score = 100 * (baseline_step_ms - variant_step_ms) / baseline_step_ms

             A score near 0% means the workload is compute-bound or insensitive
             to scale-up BW; a high score means communication on the intra-node
             fabric is on the critical path.

Unit: Percentage (%)
Returns: Float, or -1.0 if simulation failed or trace type is unsupported

Supported trace types:
  json — PyTorch-profiler Chrome JSON files (rank*_trace.json), via comm-only mode
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PIPELINE = _REPO_ROOT / "simulation" / "pipeline.py"

# Simulation defaults matching simulation/examples/13_utility_calculation.sh
_INTRA_TOPOLOGY = "Ring"
_INTER_TOPOLOGY = "Switch"
_INTRA_LAT_NS = 50
_INTER_LAT_NS = 50000
_COLLECTIVE_ALGO = "ring"
_COMPUTE_MODEL = "kernels"
_KERNEL_DEP_MODE = "rank"


def _load_yaml(directory: str) -> dict:
    for fn in os.listdir(directory):
        if fn.endswith(".yaml"):
            with open(os.path.join(directory, fn), encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def _get_trace_types(directory: str) -> list:
    return _load_yaml(directory).get("metric_source", {}).get("traces", [])


def _get_hw_params(directory: str):
    """
    Read (inter_bw_GBps, intra_bw_GBps, gpus_per_node) from the workload YAML.
    bandwidth_gbps[0] = scale-out (inter), [1] = scale-up (intra).
    Returns None if intra BW is missing or non-numeric.
    """
    cfg = _load_yaml(directory)
    hw = cfg.get("workload", {}).get("hardware", {})
    bw_list = hw.get("network_topo", {}).get("bandwidth_gbps", [])
    gpus_per_node = hw.get("xpu_spec", {}).get("count_per_node", 4)

    inter_raw = bw_list[0] if len(bw_list) > 0 else 200
    intra_raw = bw_list[1] if len(bw_list) > 1 else None

    try:
        inter_gbps = float(inter_raw)
    except (ValueError, TypeError):
        inter_gbps = 200.0

    if intra_raw is None:
        print("Error: bandwidth_gbps[1] (scale-up) missing from YAML", file=sys.stderr)
        return None
    try:
        intra_gbps = float(intra_raw)
    except (ValueError, TypeError):
        print(f"Error: bandwidth_gbps[1]={intra_raw!r} is not numeric", file=sys.stderr)
        return None

    return inter_gbps / 8.0, intra_gbps / 8.0, int(gpus_per_node)


def _run_simulation(trace_dir: str, output_dir: str,
                    inter_bw: float, intra_bw: float,
                    gpus_per_node: int,
                    reuse_from: str | None = None) -> float | None:
    """
    Invoke simulation/pipeline.py --mode comm-only.
    Returns simulated step time in ms, or None on failure.
    """
    cmd = [
        sys.executable, str(_PIPELINE),
        "--mode", "comm-only",
        "--trace-dir", trace_dir,
        "--output-dir", output_dir,
        "--gpus-per-node", str(gpus_per_node),
        "--intra-topology", _INTRA_TOPOLOGY,
        "--intra-bandwidth", f"{intra_bw:.6g}",
        "--intra-latency", str(_INTRA_LAT_NS),
        "--topology", _INTER_TOPOLOGY,
        "--bandwidth", f"{inter_bw:.6g}",
        "--latency", str(_INTER_LAT_NS),
        "--collective-algo", _COLLECTIVE_ALGO,
        "--compute-model", _COMPUTE_MODEL,
        "--kernel-dependency-mode", _KERNEL_DEP_MODE,
    ]
    if reuse_from is not None:
        cmd += ["--reuse-et-from", reuse_from]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        combined = result.stdout + result.stderr
        match = re.search(r"Simulated step time:\s*([\d.]+)\s*ms", combined)
        if match:
            return float(match.group(1))
        print(
            f"Error: 'Simulated step time' not found in pipeline output.\n"
            f"Last 2000 chars:\n{combined[-2000:]}",
            file=sys.stderr,
        )
        return None
    except subprocess.TimeoutExpired:
        print("Error: simulation timed out after 3600 s", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error running simulation: {e}", file=sys.stderr)
        return None


def _has_valid_rank_traces(directory: str) -> bool:
    """
    Return True if the directory has at least 2 rank*_trace.json files.
    Single-rank traces can't exercise scale-up bandwidth.
    """
    import glob
    rank_files = glob.glob(os.path.join(directory, "rank*_trace.json"))
    return len(rank_files) >= 2


def metric_cal(directory: str) -> float:
    """
    Compute scale-up bandwidth utility for the trace in *directory*.

    Returns:
        float: % step-time improvement from doubling intra-node BW (0–100+),
               or -1.0 if the metric cannot be computed.
    """
    trace_types = _get_trace_types(directory)
    if "json" not in trace_types:
        print(
            f"Error: scale_up_bw_utility requires json trace type, "
            f"got {trace_types}",
            file=sys.stderr,
        )
        return -1.0

    if not _has_valid_rank_traces(directory):
        print(
            "Error: scale_up_bw_utility requires ≥2 rank*_trace.json files "
            "with GPU kernel events (cat='kernel')",
            file=sys.stderr,
        )
        return -1.0

    hw = _get_hw_params(directory)
    if hw is None:
        return -1.0

    inter_bw, intra_bw, gpus_per_node = hw
    double_intra_bw = intra_bw * 2.0

    tmpdir = tempfile.mkdtemp(prefix="ccl_bench_scaleup_")
    try:
        base_out = os.path.join(tmpdir, "base")
        variant_out = os.path.join(tmpdir, "scaleup_bw_2x")

        print(
            f"Baseline:  inter={inter_bw:.4g} GB/s  intra={intra_bw:.4g} GB/s  "
            f"gpus_per_node={gpus_per_node}"
        )
        base_step = _run_simulation(
            directory, base_out, inter_bw, intra_bw, gpus_per_node
        )
        if base_step is None:
            return -1.0

        print(f"Variant:   inter={inter_bw:.4g} GB/s  intra={double_intra_bw:.4g} GB/s (2×)")
        variant_step = _run_simulation(
            directory, variant_out, inter_bw, double_intra_bw, gpus_per_node,
            reuse_from=base_out,
        )
        if variant_step is None:
            return -1.0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if base_step <= 0:
        print("Error: baseline step time is zero or negative", file=sys.stderr)
        return -1.0

    utility = 100.0 * (base_step - variant_step) / base_step
    print(
        f"Baseline step: {base_step:.3f} ms  |  2× intra-BW step: {variant_step:.3f} ms  "
        f"|  Scale-up BW utility: {utility:.4f}%"
    )
    return float(utility)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scale_up_bw_utility.py <trace_directory>")
        sys.exit(1)
    result = metric_cal(sys.argv[1])
    print(result)
