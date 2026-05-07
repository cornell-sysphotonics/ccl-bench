# CCL-Bench

![CCL-Bench logo](./assets/logo.png)

CCL-Bench is a trace-based benchmark for LLM systems. Each benchmark row is backed by workload metadata and profiler artifacts, so results can be recomputed, audited, and extended as new models, frameworks, hardware, and collective communication libraries are added.

The project is organized around three layers:

- **Evidence**: workload cards, run metadata, and external profiler traces.
- **Analysis**: metric tools that consume trace directories and return leaderboard values.
- **Presentation**: a static website generated from configured trace and metric pairs.

Raw traces are intentionally not committed to git. They are large artifacts and should live in shared storage such as `/data/ccl-bench_trace_collection` or an external artifact store. The repository keeps lightweight metadata, scripts, metric code, and generated website data.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `workload_card_template.yaml` | Canonical metadata template for benchmark rows. |
| `trace_collection/` | Lightweight workload cards and trace collection notes. |
| `trace_gen/` | Guidance and helpers for collecting profiler traces. |
| `tools/` | Metric toolkit. Each metric is implemented as an importable tool. |
| `website/` | Static leaderboard and generated benchmark data. |
| `workloads/` | Standard workload definitions used to select model and parallelism targets. |
| `scripts/` | Reproducibility and collection scripts for specific systems or experiments. |
| `agent/` | Experimental/private config tuning agents. |
| `simulation/` | Experimental/private trace-based simulation utilities. |

## Quick Start

Create a local environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run one metric on a trace directory:

```bash
python tools/main.py --trace /path/to/trace_dir --metric avg_step_time
```

Regenerate the static website data after adding or changing configured traces:

```bash
python website/generate_data.py
cd website
python -m http.server 8081
```

Then open `http://localhost:8081`.

## Adding A Benchmark Row

1. Select a standard workload from `workloads/` or `trace_collection/workload.md`.
2. Collect profiler artifacts outside the repository. Keep the final trace directory name stable.
3. Fill in `workload_card_template.yaml` and store the card with the trace artifacts.
4. Add the lightweight workload card under `trace_collection/<workload_name>/` when it is useful for review and reproducibility.
5. Add the trace and metric mapping to `website/benchmark_config.json`.
6. Regenerate `website/benchmark_data.json` and `website/data.js`.

Each row should make clear:

- model, phase, precision, dataset, batch size, and sequence lengths;
- hardware type, GPU/TPU count, and per-node count;
- framework and compiler/runtime versions;
- tensor/data/pipeline/expert parallelism;
- communication library and relevant environment variables;
- which trace artifacts were used for each metric.

## Metrics

Metrics are implemented in `tools/` and invoked through `tools/main.py`. The public website uses the subset configured in `website/benchmark_config.json`; additional tools can remain in the repository for experiments as long as they are documented and do not require checked-in raw traces.

See `tools/README.md` for the supported metric interface and current dashboard metrics.

## Artifact Policy

Commit:

- source code and scripts required to reproduce a row;
- workload cards and small metadata files;
- generated website JSON/JS when updating the public leaderboard;
- documentation explaining non-obvious trace or environment requirements.

Do not commit:

- virtual environments or package caches;
- raw profiler dumps unless they are intentionally tiny test fixtures;
- local API keys, credentials, or machine-specific scratch paths;
- large intermediate logs that are not part of the artifact.

## Trace Storage Location

The canonical shared trace directory on Perlmutter is `/data/ccl-bench_trace_collection`.
This path appears in three places and must be updated consistently if you move traces
to a different mount point or machine:

| Location | How to change |
| --- | --- |
| `website/benchmark_config.json` — every `"trace":` path | Update each path prefix to match your local mount point. The paths must resolve on whichever machine runs `python website/generate_data.py`. |
| `agent/ccl_bench_agent/tuning_config.yaml` — `publish_dir` | Set `publish_dir` to the desired destination. CCL-Search copies per-iteration traces there. Leave empty to skip publishing. |
| `README.md` — example path in the intro paragraph | Documentation only; no functional effect. |

If you are running on a different cluster, set `publish_dir` in `tuning_config.yaml` and
update the `"trace":` paths in `benchmark_config.json` before regenerating the website.

## Development Notes

CCL-Search (`agent/`) and the simulation pipeline (`simulation/`) are first-class
contributions: CCL-Search automates configuration tuning and records every trial as a
benchmark entry; the simulation pipeline converts traces to Chakra execution graphs for
Astra-Sim what-if analysis. Both require the shared trace directory to be accessible.
