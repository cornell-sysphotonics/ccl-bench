# Trace Collection Metadata

`trace_collection/` stores lightweight benchmark metadata. It is not the primary storage location for raw profiler traces.

For each benchmark row, keep the raw artifacts in shared storage or external artifact storage, then keep the corresponding workload card and small notes in this repository when they are useful for review.

## Expected Row Structure

A trace directory outside git normally looks like:

```text
<workload_name>/
  <workload_name>.yaml        # workload card
  rank0_trace.json            # optional Kineto/XLA trace
  *.pt.trace.json             # optional Kineto rank traces
  *.sqlite                    # optional Nsight Systems export
  *.nsys-rep                  # optional Nsight Systems report
  *.latency.json              # optional benchmark latency output
  README.md                   # optional run notes
```

Inside this repository, prefer committing only:

```text
trace_collection/<workload_name>/
  <workload_name>.yaml
  README.md                   # optional, small
  workload.toml               # optional training config, if needed to reproduce
```

## Workload Cards

Every benchmark row should have a workload card based on `../workload_card_template.yaml`. The card is the contract between trace collection, metric tools, and the website.

Important fields include:

- workload name and phase;
- model architecture and parameter counts;
- batch size, sequence length, input length, and output length;
- hardware type, hardware model, total device count, and devices per node;
- framework and compiler/runtime versions;
- tensor/data/pipeline/expert parallelism;
- communication library and relevant environment variables;
- trace types available for metric extraction.

Use stable workload names. A good name includes model, framework, key parallelism, batch/sequence settings, communication library, and platform.

## Raw Artifact Storage

Raw traces are large and should not be committed. For local deployments, the website generator expects configured trace directories to be available from paths such as:

```text
/data/ccl-bench_trace_collection/<workload_name>
```

When adding a row to the website:

1. Copy the raw trace directory and workload card to shared storage.
2. Add the trace path and metric list to `../website/benchmark_config.json`.
3. Regenerate `../website/benchmark_data.json` and `../website/data.js`.
4. Commit the lightweight metadata and generated website files.

## Standard Workloads

`workload.md` tracks the standard workload set used for cross-system comparisons. When possible, new traces should reuse existing model type, batch size, sequence length, and parallelism so rows remain comparable across hardware and communication libraries.

## What Not To Commit

Do not commit:

- virtual environments;
- package caches;
- large raw profiler dumps;
- temporary logs;
- credentials or local machine secrets.

Small configs, workload cards, and concise run notes are encouraged.
