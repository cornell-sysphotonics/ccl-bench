#!/usr/bin/env python3
"""Run all applicable tools on an uploaded trace directory and write results.json.

Usage:
    python website/process_upload.py --trace-dir <dir> --output <results.json>

Must be run from the repository root so tools/main.py is resolvable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = REPO_ROOT / "website" / "benchmark_config.json"


def _extract_archives(trace_dir: Path) -> None:
    """Extract ZIP/tar archives and decompress .gz trace files."""
    import gzip as _gzip
    for f in list(trace_dir.iterdir()):
        try:
            if f.suffix == ".zip":
                print(f"Extracting {f.name}", file=sys.stderr)
                with zipfile.ZipFile(f) as zf:
                    zf.extractall(trace_dir)
                f.unlink()
            elif f.name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar.zst", ".tar")):
                print(f"Extracting {f.name}", file=sys.stderr)
                with tarfile.open(f) as tf:
                    tf.extractall(trace_dir)
                f.unlink()
            elif f.name.endswith(".json.gz"):
                # Decompress so tools that only read .json can find the file
                out = trace_dir / f.name[:-3]  # strip .gz
                print(f"Decompressing {f.name} -> {out.name}", file=sys.stderr)
                with _gzip.open(f, "rb") as gz, open(out, "wb") as dest:
                    dest.write(gz.read())
                f.unlink()
        except Exception as e:
            print(f"Warning: failed to extract {f.name}: {e}", file=sys.stderr)

    # Flatten single-directory archives (e.g. foo.zip -> foo/ -> contents)
    entries = [e for e in trace_dir.iterdir() if not e.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        for child in entries[0].iterdir():
            child.rename(trace_dir / child.name)
        entries[0].rmdir()


def _count_iterations(trace_dir: Path) -> int | None:
    """Count training iterations from trace events."""
    import statistics as _stats
    for f in trace_dir.iterdir():
        if f.suffix != ".json" or not f.is_file():
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            events = data.get("traceEvents", []) if isinstance(data, dict) else []
        except Exception:
            continue
        # TorchXLA: jit_train_step events per device
        jit = [e for e in events if e.get("ph") == "X" and isinstance(e.get("name"), str)
               and e["name"].startswith("jit_train_step")]
        if jit:
            first_pid = min(e["pid"] for e in jit)
            n = len([e for e in jit if e["pid"] == first_pid])
            print(f"Detected {n} iterations from jit_train_step", file=sys.stderr)
            return n
        # MaxText: count large inter-StepMarker gaps
        sm = sorted([e for e in events if e.get("ph") == "X" and e.get("name") == "StepMarker"],
                    key=lambda e: e["ts"])
        if len(sm) >= 2:
            gaps = [sm[i+1]["ts"] - sm[i]["ts"] for i in range(len(sm)-1)]
            med = _stats.median(gaps)
            n = sum(1 for g in gaps if g > med * 3)
            if n:
                print(f"Detected {n} iterations from StepMarker", file=sys.stderr)
                return n
    return None


def _detect_trace_type(trace_dir: Path) -> str:
    """Return a trace_type string based on file contents of the directory."""
    files = list(trace_dir.iterdir())
    names = [f.name for f in files]
    if any(".trace.json" in n for n in names):
        return "json_tpu"
    if any(n.endswith(".nsys-rep") for n in names):
        return "nsys"
    # Generic JSON — peek at the first event to distinguish TPU vs GPU
    for f in files:
        if f.suffix == ".json" or f.name.endswith(".json.gz"):
            try:
                import gzip, json as _json
                opener = gzip.open if f.name.endswith(".gz") else open
                with opener(f, "rt") as fh:
                    data = _json.load(fh)
                events = data.get("traceEvents", [])
                names_set = {e.get("name", "") for e in events[:500]}
                if any("jit_train_step" in n or "StepMarker" in n or "core.py" in n for n in names_set):
                    return "json_tpu"
                if any("ProfilerStep" in n or "cudaLaunchKernel" in n for n in names_set):
                    return "json"
            except Exception:
                pass
    return "json"


def _ensure_yaml(trace_dir: Path, trace_type: str) -> None:
    """Create a minimal YAML with the detected trace type so tools can route correctly."""
    try:
        import yaml
    except ImportError:
        return
    xpu_type = "TPU" if trace_type == "json_tpu" else "GPU"
    # Check if an existing YAML already has correct values; if so, leave it alone
    for f in sorted(trace_dir.glob("*.yaml")):
        try:
            with open(f) as fh:
                existing = yaml.safe_load(fh)
            ms = (existing or {}).get("metric_source", {}) or {}
            hw = ((existing or {}).get("workload", {}) or {}).get("hardware", {}) or {}
            xpu = hw.get("xpu_spec", {}) or {}
            existing_traces = set(ms.get("traces") or [])
            existing_type = (xpu.get("type") or "").lower()
            # Patch in-place if needed
            if not existing_traces or existing_type != xpu_type.lower():
                print(f"Patching YAML: traces={[trace_type]}, xpu_type={xpu_type}", file=sys.stderr)
                existing.setdefault("metric_source", {})["traces"] = [trace_type]
                existing.setdefault("workload", {}).setdefault("hardware", {}).setdefault("xpu_spec", {})["type"] = xpu_type
            model = existing.setdefault("workload", {}).setdefault("model", {})
            if not model.get("iteration"):
                n = _count_iterations(trace_dir)
                if n:
                    model["iteration"] = n
            f.write_text(yaml.dump(existing, default_flow_style=False, allow_unicode=True))
            return
        except Exception:
            pass
    # No YAML at all — write a minimal one
    n = _count_iterations(trace_dir)
    minimal = {
        "version": 1,
        "workload": {
            "hardware": {"xpu_spec": {"type": xpu_type}},
            "model": {"iteration": n} if n else {},
        },
        "metric_source": {"traces": [trace_type]},
    }
    (trace_dir / "_autodetected.yaml").write_text(
        yaml.dump(minimal, default_flow_style=False, allow_unicode=True)
    )
    print(f"Created minimal YAML: xpu_type={xpu_type}, traces={[trace_type]}", file=sys.stderr)


def _pick_metrics(metric_info: dict) -> list[str]:
    return list(metric_info.keys())


def _run_metric(trace_dir: str, metric: str) -> float | str | None:
    try:
        r = subprocess.run(
            [sys.executable, "tools/main.py", "--trace", trace_dir, "--metric", metric],
            capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
        )
        if r.returncode != 0:
            print(f"Metric {metric} failed: {r.stderr.strip()[-200:]}", file=sys.stderr)
            return None
        lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
        if not lines:
            return None
        raw = lines[-1]
        try:
            v = float(raw)
            return None if (v != v or v == -1.0) else float(f"{v:.6g}")
        except ValueError:
            return raw
    except Exception as e:
        print(f"Metric {metric} error: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir).resolve()
    output_path = Path(args.output)

    _extract_archives(trace_dir)

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    metric_info = config.get("metric_info", {})

    trace_type = _detect_trace_type(trace_dir)
    print(f"Detected trace type: {trace_type}", file=sys.stderr)

    # Some tools route by checking if "tpu" appears in the directory name.
    # Uploaded directories have UUID names, so move files into a subdirectory
    # that satisfies the naming convention.
    if trace_type == "json_tpu" and "tpu" not in trace_dir.name.lower():
        tpu_subdir = trace_dir / "tpu_trace"
        tpu_subdir.mkdir(exist_ok=True)
        for f in list(trace_dir.iterdir()):
            if f.name != "tpu_trace":
                f.rename(tpu_subdir / f.name)
        trace_dir = tpu_subdir
        print(f"Moved files to {trace_dir} for TPU tool routing", file=sys.stderr)

    _ensure_yaml(trace_dir, trace_type)
    metrics = _pick_metrics(metric_info)

    results: dict[str, object] = {}
    if metrics:
        with ThreadPoolExecutor(max_workers=min(8, len(metrics))) as pool:
            futures = {pool.submit(_run_metric, str(trace_dir), m): m for m in metrics}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()

    computed = {m: v for m, v in results.items() if v is not None}
    print(f"Computed {len(computed)}/{len(metrics)} metrics: {list(computed.keys())}", file=sys.stderr)

    output_path.write_text(json.dumps({
        "metrics": results,
        "metric_info": {m: metric_info[m] for m in results if m in metric_info},
    }, indent=2))


if __name__ == "__main__":
    main()
