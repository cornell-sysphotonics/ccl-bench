#!/usr/bin/env python3
"""
Generate benchmark data for the CCL-Bench website.

Run from the REPOSITORY ROOT (not from website/):
    python website/generate_data.py

Outputs:
    website/data.js             — Embedded JS for use directly in the browser (no server needed)
    website/benchmark_data.json — Machine-readable JSON

Prerequisites:
    pip install pyyaml

Incremental updates (via benchmark_config.json flags):
    - Set "required_update": true on a trace entry  → recompute all its metrics.
    - Set "required_update": true on a metric_info entry → recompute that metric
      across every trace that lists it.
    - Entries with "required_update": false reuse cached values from the existing
      benchmark_data.json (if present) and are skipped entirely.
    - New traces/metrics (not yet in the cache) are always computed.
"""

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML not found. Install with: pip install pyyaml")
    sys.exit(1)


# ── YAML helpers ──────────────────────────────────────────────────────────────

def find_yaml(trace_dir: str) -> dict | None:
    """Find and parse the workload card YAML in the trace directory."""
    trace_path = Path(trace_dir)
    print(trace_path)
    yaml_files = sorted(trace_path.glob("*.yaml"))
    if not yaml_files:
        print(f"  Warning: No YAML found in {trace_dir}")
        return None
    try:
        with open(yaml_files[0]) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"  Warning: YAML parse error ({yaml_files[0].name}): {e}")
        return None


def extract_metadata(yaml_data: dict | None, trace_dir: str) -> dict:
    """Flatten key workload-card fields into a display-friendly dict."""
    meta: dict = {"workload_name": Path(trace_dir).name}
    if not yaml_data:
        return meta

    w    = yaml_data.get("workload", {}) or {}
    mod  = w.get("model", {}) or {}
    dat  = w.get("data", {}) or {}
    hw   = w.get("hardware", {}) or {}
    xpu  = hw.get("xpu_spec", {}) or {}
    net  = hw.get("network_topo", {}) or {}
    ex   = yaml_data.get("Model-executor", {}) or {}
    fw   = ex.get("framework", {}) or {}
    par  = ex.get("model_plan_parallelization", {}) or {}
    cl   = ex.get("communication_library", {}) or {}
    ms   = yaml_data.get("metric_source", {}) or {}

    bw = net.get("bandwidth_gbps") or []

    meta.update({
        # Top-level
        "description":        (yaml_data.get("description") or "").strip(),
        "hf_url":             yaml_data.get("hf_url") or "",
        "trace_url":          yaml_data.get("trace_url") or "",
        # Model
        "model_family":       mod.get("model_family") or "",
        "phase":              mod.get("phase") or "",
        "precision":          mod.get("precision") or "",
        "moe":                mod.get("moe", False),
        "granularity":        mod.get("granularity") or "",
        "epochs":             mod.get("epochs") or "",
        "iteration":          mod.get("iteration") or "",
        # Data
        "batch_size":         dat.get("batch_size") or "",
        "seq_len":            dat.get("seq_len") or "",
        "input_len":          dat.get("input_len") or "",
        "output_len":         dat.get("output_len") or "",
        "dataset":            dat.get("dataset") or "",
        # Hardware — XPU
        "hardware_type":      xpu.get("type") or "",
        "hardware_model":     xpu.get("model") or "",
        "total_count":        xpu.get("total_count") or "",
        "count_per_node":     xpu.get("count_per_node") or "",
        # Hardware — network
        "network_topology":   net.get("topology") or "",
        "bandwidth_scaleout": bw[0] if len(bw) > 0 else "",
        "bandwidth_scaleup":  bw[1] if len(bw) > 1 else "",
        # Hardware — driver
        "driver_version":     hw.get("driver_version") or "",
        # Framework
        "framework":          fw.get("name") or "",
        "compiler":           fw.get("compiler_tool_selection") or "",
        # Parallelism
        "tp":                 par.get("tp") or "",
        "pp":                 par.get("pp") or "",
        "pp_mb":              par.get("pp_mb") or "",
        "dp_replicate":       par.get("dp_replicate") or "",
        "dp_shard":           par.get("dp_shard") or "",
        "ep":                 par.get("ep") or "",
        "cp":                 par.get("cp") or "",
        # Communication library
        "comm_library":       cl.get("name") or "",
        "comm_library_ver":   cl.get("version") or "",
        "comm_env":           cl.get("env") or {},
        # Protocol selection
        "protocol_selection": ex.get("protocol_selection") or [],
        # Metric source
        "trace_types":        ms.get("traces") or [],
        "metric_traces":      ms.get("metrics_specific_trace") or [],
    })
    return meta


# ── Metric runner ─────────────────────────────────────────────────────────────

def run_metric(trace_dir: str, metric: str) -> float | str | None:
    """
    Invoke tools/main.py and extract the scalar result.
    The tool may print debug lines before the actual value; we take the last line.
    """
    try:
        result = subprocess.run(
            [sys.executable, "tools/main.py", "--trace", trace_dir, "--metric", metric],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0:
            snippet = result.stderr.strip()[:300]
            print(f"\n    [stderr] {snippet}")
            return None

        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return None

        raw = lines[-1]
        try:
            val = float(raw)
            if val != val:          # NaN → treat as missing
                return None
            if val == -1.0:         # sentinel used by group_9 tools for "data unavailable"
                return None
            return float(f"{val:.6g}")
        except ValueError:
            return raw              # return raw string for non-numeric metrics

    except subprocess.TimeoutExpired:
        print("    [timeout after 1800s]")
        return None
    except Exception as e:
        print(f"    [error] {e}")
        return None


# ── Cache helpers ──────────────────────────────────────────────────────────────

def load_cache(json_path: Path) -> dict[str, dict]:
    """
    Load existing benchmark_data.json and return a mapping
    { trace_dir: { metric: value, ... } }.
    Returns an empty dict if the file does not exist or cannot be parsed.
    """
    if not json_path.exists():
        return {}
    try:
        with open(json_path) as f:
            data = json.load(f)
        return {
            row["trace"]: {
                k: (None if v == -1.0 else v)
                for k, v in row.get("metrics", {}).items()
            }
            for row in data.get("rows", [])
        }
    except Exception as e:
        print(f"  Warning: could not load cache from {json_path}: {e}")
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not Path("tools/main.py").exists():
        print("Error: Run this script from the repository root (where tools/main.py lives).")
        sys.exit(1)

    config_path = Path("website/benchmark_config.json")
    if not config_path.exists():
        print(f"Error: {config_path} not found.")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    pairs             = config.get("pairs", [])
    metric_info       = config.get("metric_info", {})
    metric_categories = config.get("metric_categories", [])

    # Metrics whose implementation changed → must recompute across all traces.
    stale_metrics: set[str] = {
        m for m, info in metric_info.items() if info.get("required_update", False)
    }
    if stale_metrics:
        print(f"Metrics marked required_update: {sorted(stale_metrics)}")

    json_path = Path("website/benchmark_data.json")
    cache = load_cache(json_path)
    if cache:
        print(f"Loaded cache with {len(cache)} trace(s) from {json_path}")

    all_metrics: set = set()

    # ── Phase 1: resolve metadata and split work into compute vs cache ─────────
    trace_plans = []   # list of dicts describing each trace's work
    work_items  = []   # (trace_dir, metric) pairs that need fresh computation

    for i, entry in enumerate(pairs):
        trace_dir        = entry["trace"]
        metrics_spec     = entry.get("metrics", "auto")
        trace_needs_full = entry.get("required_update", True)
        name             = Path(trace_dir).name

        print(f"[{i+1}/{len(pairs)}] Resolving {name}")

        yaml_data = find_yaml(trace_dir)
        metadata  = extract_metadata(yaml_data, trace_dir)

        if metrics_spec == "auto":
            trace_types = set(metadata.get("trace_types", []))
            if not trace_types:
                print(f"  Warning: no trace_types in YAML — skipping auto-selection")
                metrics = []
            else:
                metrics = [
                    m for m, info in metric_info.items()
                    if set(info.get("trace_types", [])) & trace_types
                    and (
                        not info.get("phases")
                        or str(metadata.get("phase", "")).lower()
                        in {str(p).lower() for p in info.get("phases", [])}
                    )
                ]
                print(f"  Auto-selected {len(metrics)} metrics for {sorted(trace_types)}: "
                      f"{', '.join(metrics)}")
        else:
            metrics = metrics_spec

        cached_metrics = cache.get(trace_dir, {})
        to_compute = []
        to_cache   = []

        for metric in metrics:
            all_metrics.add(metric)
            needs_compute = (
                trace_needs_full
                or metric in stale_metrics
                or metric not in cached_metrics
            )
            if needs_compute:
                to_compute.append(metric)
                work_items.append((trace_dir, metric))
            else:
                to_cache.append(metric)

        trace_plans.append({
            "trace_dir":      trace_dir,
            "metadata":       metadata,
            "cached_metrics": cached_metrics,
            "to_compute":     to_compute,
            "to_cache":       to_cache,
        })

    # ── Phase 2: parallel metric computation ──────────────────────────────────
    computed: dict[tuple[str, str], object] = {}  # (trace_dir, metric) → value

    if work_items:
        max_workers = min(32, len(work_items))
        print(f"\nComputing {len(work_items)} (trace, metric) pair(s) "
              f"with up to {max_workers} parallel workers …\n")

        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for trace_dir, metric in work_items:
                fut = executor.submit(run_metric, trace_dir, metric)
                futures[fut] = (trace_dir, metric)

            for fut in as_completed(futures):
                trace_dir, metric = futures[fut]
                val = fut.result()
                computed[(trace_dir, metric)] = val
                name = Path(trace_dir).name
                print(f"  ✓ {name}  {metric} → {val}")

    # ── Phase 3: assemble rows ────────────────────────────────────────────────
    rows = []
    for plan in trace_plans:
        trace_dir      = plan["trace_dir"]
        cached_metrics = plan["cached_metrics"]
        metric_results: dict = {}

        for metric in plan["to_cache"]:
            metric_results[metric] = cached_metrics[metric]

        for metric in plan["to_compute"]:
            metric_results[metric] = computed.get((trace_dir, metric))

        rows.append({
            "trace":    trace_dir,
            "metadata": plan["metadata"],
            "metrics":  metric_results,
        })

    output = {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "all_metrics":       sorted(all_metrics),
        "metric_categories": metric_categories,
        "metric_info":       metric_info,
        "rows":              rows,
    }

    # ── Write JSON ────────────────────────────────────────────────────────────
    json_path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ {json_path}  ({len(rows)} rows, {len(all_metrics)} metrics)")

    # ── Write data.js ─────────────────────────────────────────────────────────
    js_path = Path("website/data.js")
    js_path.write_text(
        "// Auto-generated by website/generate_data.py — do not edit manually\n"
        f"const BENCHMARK_DATA = {json.dumps(output, indent=2)};\n"
    )
    print(f"✓ {js_path}")

    # ── Refresh inline CSS in index.html ──────────────────────────────────────
    html_path = Path("index.html")
    html = html_path.read_text()
    css = Path("website/styles.css").read_text()
    css_block = f"<!-- STYLES_START -->\n<style>\n{css}</style>\n<!-- STYLES_END -->"
    updated = re.sub(
        r"<!-- STYLES_START -->.*?<!-- STYLES_END -->",
        lambda _: css_block,
        html, flags=re.DOTALL,
    )
    if updated != html:
        html_path.write_text(updated)
        print(f"✓ {html_path}  (CSS refreshed)")

    print()
    print("Open the website:")
    print("  open index.html          # macOS")
    print("  xdg-open index.html      # Linux")


if __name__ == "__main__":
    main()
