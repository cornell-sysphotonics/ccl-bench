# LLM Collectives Profiler – NSys Analysis Toolkit

This repository contains a lightweight analysis tool used to extract and summarize
communication and performance metrics from **Nsight Systems (NSys)** traces generated
during distributed LLM training runs. The tool is designed to work directly with
TorchTitan training traces and produce a single human-readable text report per run.

The focus of this project is understanding **communication behavior**, **overlap with
compute**, and **network utilization** under different parallelism configurations
(data parallelism and tensor parallelism).

---

## Tool Development

The analysis pipeline is implemented in Python and is based on parsing NSys JSON trace
files generated during training. Rather than building a new profiler from scratch,
we extend the NSys workflow by post-processing its traces to extract higher-level,
distributed training metrics.

The tool was originally implemented as a single script (`nsys_analyzer.py`) and later
refactored into multiple files for clarity and maintainability. Importantly, this
refactor preserves **all original logic and metrics**, and the final output text file
remains identical in content.

Key design goals:
- minimal assumptions about the training framework
- compatible with TorchTitan NSys traces
- single command to generate a full performance report
- easy comparison across different parallelism settings

---

## Pipeline Overview

At a high level, the analysis pipeline consists of three stages:

### 1. Trace Collection (Training Side)

During training, NSys is enabled via TorchTitan’s profiling configuration. Each GPU
(rank) produces a JSON trace file containing:
- kernel execution events
- NCCL collective operations
- timestamps and durations
- metadata about tensor sizes and data types

These per-rank JSON traces are stored in a directory such as:

```
outputs/profile_trace/llama3_8B_dp2_tp2_ngpu4/iteration_10/
```

Each directory contains one JSON file per rank.

---

### 2. Trace Parsing and Metric Extraction

The analysis tool:
- loads all JSON trace files in the provided directory
- classifies events into **communication** (NCCL collectives) and **compute**
- infers message sizes for communication operations
- aggregates metrics across all ranks

All parsing and metric calculations are deterministic and based entirely on NSys data.

---

### 3. Report Generation

After processing all traces, the tool writes a single text report containing:
- throughput metrics
- communication–computation overlap
- network bandwidth estimates
- traffic breakdown by collective type

The report is written to:

```
trace_analysis/<timestamp>_<model>_dpXxY_tpZ_gpuN.txt
```

This makes it easy to archive and compare results across runs.

---

## How to Run the Analyzer (Examples)

### Basic usage

From the **root of the repository**, run:

```
python tools/main.py torchtitan/outputs/profile_trace/llama3_8B_dp2_tp2_ngpu4/iteration_10 --config torchtitan/torchtitan/models/llama3/train_configs/llama3_8b.toml
```
