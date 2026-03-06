# Load Imbalance Experiment

This directory contains scripts and tools for running MoE (Mixture-of-Experts) load imbalance experiments, benchmarking the performance of Baseline (Triton MoE) vs DefaultAll2All (FlashInfer MoE) implementations under different input workload characteristics.

## Overview

The experiment measures how input data characteristics affect MoE routing imbalance and correlates this with performance metrics (TTFT, TPOT, throughput). Datasets with varying predicted imbalance scores are generated and tested against both Baseline and DefaultAll2All implementations.

## File Descriptions

### Server Launch Scripts

- **`launch_baseline.sh`**: Launches the Baseline vLLM server with Triton MoE kernel
  - Uses Tensor Parallelism (TP=4)
  - Configurable: `MODEL`, `PORT`, `BATCH_TOKENS`, `MAX_SEQS`, `MAX_MODEL_LEN`
  - Default: 4 GPUs, TP=4, max-num-batched-tokens=8192, max-num-seqs=32

- **`launch_defaultAll2All.sh`**: Launches the DefaultAll2All vLLM server with FlashInfer MoE backend
  - Uses Expert Parallelism (EP=4)
  - Configurable: `MODEL`, `PORT`, `BATCH_TOKENS`, `MAX_SEQS`, `MAX_MODEL_LEN`, `EP_SIZE`
  - Default: 4 GPUs, EP=4, max-num-batched-tokens=8192, max-num-seqs=32

### Client and Benchmarking

- **`client.sh`**: Runs benchmark client against the server
  - Waits for server to be ready (checks `/health` endpoint)
  - Sends requests from specified dataset
  - Saves results to `results_json/` directory
  - Arguments: `DATASET_PATH`, `INPUT_SEQUENCE_LEN` (optional), `PORT` (optional)

- **`benchmark.py`**: Main benchmarking script that:
  - Loads datasets from JSONL files
  - Sends async requests to vLLM server
  - Collects performance metrics (TTFT, TPOT, throughput)
  - Computes predicted imbalance scores using `ImbalancePredictor`
  - Saves results with imbalance predictions to JSON files

- **`bench_utils.py`**: Utility functions for benchmarking
  - Async request handling
  - Request/response parsing
  - Metric calculation helpers

### Dataset Generation

- **`generate_datasets.py`**: Generates synthetic datasets with varying imbalance characteristics
  - Uses domain queries from `domain_queries.json`
  - Creates 50 datasets with different imbalance-inducing strategies
  - Saves datasets as `{index}_{score:.4f}.jsonl` in `datasets/` directory
  - Generates imbalance distribution plots

- **`domain_queries.json`**: Contains domain-specific queries (25 domains, ~50 queries each)
  - Used as source material for dataset generation
  - Domains include: programming, science, cooking, sports, music, travel, etc.

### Imbalance Analysis

- **`get_real_imbalance.py`**: Computes real MoE load imbalance from gate logs (per-step method)
  - Reads `gate_logs_*.json` files from a directory
  - Calculates Coefficient of Variation (CV) **per step**, then averages across steps
  - Processes each inference step independently
  - Returns imbalance score (lower = more balanced)

- **`compute_all_real_imbalance.py`**: Batch processes multiple gate log directories (per-step method)
  - Traverses subdirectories (e.g., `gates_logs_01`, `gates_logs_02`, ...)
  - Computes CV per step then averages (same method as `get_real_imbalance.py`)
  - Saves results to JSON file mapping directory names to CV scores

- **`new_real_imbalance.py`**: Computes real imbalance using **aggregated expert assignment** method
  - Reads `gate_logs_*.json` files from a single directory
  - **Aggregates all expert loads across all steps first**, then computes one CV on the totals
  - Filters out steps with >1000 tokens (removes warmup/profiling batches)
  - Also computes max/min ratio for additional insight
  - Provides detailed statistics (total tokens, expert loads breakdown)

- **`new_compute_all_real_imbalance.py`**: Batch processes using **aggregated expert assignment** method
  - Traverses subdirectories (e.g., `gates_logs_01`, `gates_logs_02`, ...)
  - Uses aggregated method: sums expert loads across all steps, then computes CV on aggregated totals
  - Filters steps with >1000 tokens
  - Saves results to JSON file

**Difference between methods:**
- **Old method** (`get_real_imbalance.py`, `compute_all_real_imbalance.py`): Computes CV per inference step, then averages the CVs. Captures step-to-step variability.
- **New method** (`new_real_imbalance.py`, `new_compute_all_real_imbalance.py`): Aggregates all expert loads across the entire experiment first, then computes one CV value. Captures overall imbalance across the entire workload.

### Visualization

- **`plot_metrics_vs_real_imbalance.py`**: Plots performance metrics vs real load imbalance (CV)
  - X-axis: Real MoE load imbalance (Coefficient of Variation)
  - Y-axis: Performance metric (TTFT, TPOT, or throughput)
  - Compares Baseline vs DefaultAll2All implementations
  - Uses scatter plots with IQR-based outlier filtering

- **`plot_metrics_vs_imbalance_score.py`**: Plots performance metrics vs predicted imbalance score
  - X-axis: Predicted imbalance score from `ImbalancePredictor`
  - Y-axis: Performance metric (TTFT, TPOT, or throughput)
  - Compares Baseline vs DefaultAll2All implementations
  - Uses scatter plots with IQR-based outlier filtering

### Data Directories

- **`datasets/`**: Contains generated JSONL dataset files
  - Format: `{index}_{predicted_imbalance_score}.jsonl`
  - Each file contains queries/prompts for benchmarking

- **`baseline_results_json/`**: Benchmark results from Baseline runs
  - One JSON file per dataset (named by dataset index)
  - Contains performance metrics and imbalance predictions

- **`default_all2all_results_json/`**: Benchmark results from DefaultAll2All runs

- **`baseline_gates_logs/`**: Gate logs from Baseline experiments
  - Subdirectories: `gates_logs_01/`, `gates_logs_02/`, etc.
  - Each contains `gate_logs_*.json` files with expert load data

- **`default_all2all_gates_logs/`**: Gate logs from DefaultAll2All experiments

## How to Run Experiments

### Prerequisites

1. Environment setup (load conda environment, install dependencies)
2. vLLM installed with MoE support
3. Model downloaded (e.g., `mistralai/Mixtral-8x7B-v0.1`)

### Step 1: Generate Datasets (if needed)

If you need to regenerate datasets with different characteristics:

```bash
python generate_datasets.py
```

This creates 50 datasets in `datasets/` directory with varying imbalance scores.

### Step 2: Start the Server

#### Option A: Baseline Server

In one terminal on the compute node:

```bash
./launch_baseline.sh [MODEL] [PORT] [BATCH_TOKENS] [MAX_SEQS] [MAX_MODEL_LEN]
```

Example:
```bash
./launch_baseline.sh mistralai/Mixtral-8x7B-v0.1 8000 2048 32 4096
```

#### Option B: DefaultAll2All Server

In one terminal on the compute node:

```bash
./launch_defaultAll2All.sh [MODEL] [PORT] [BATCH_TOKENS] [MAX_SEQS] [MAX_MODEL_LEN] [EP_SIZE]
```

Example:
```bash
./launch_defaultAll2All.sh mistralai/Mixtral-8x7B-v0.1 8000 8192 32 4096 4
```

The server will start and listen on the specified port (default: 8000).

### Step 3: Run Benchmark with Different Datasets

In another terminal on the **same node**, run the client for each dataset:

```bash
./client.sh datasets/001_0.7585.jsonl
./client.sh datasets/002_0.4321.jsonl
./client.sh datasets/003_0.4248.jsonl
# ... continue for all datasets
```

Or use a loop:

```bash
for dataset in datasets/*.jsonl; do
    ./client.sh "$dataset"
    sleep 5  # Optional: brief pause between runs
done
```

Results will be saved to `results_json/` directory, with filenames matching the dataset index (e.g., `001.json`, `002.json`).

### Step 4: Analyze Results

#### Compute Real Load Imbalance

To compute real imbalance from gate logs, you can use either method:

**Method 1: Per-step averaging (older method)**
```bash
# For baseline
python compute_all_real_imbalance.py baseline_gates_logs baseline_real_imbalance.json

# For defaultAll2All
python compute_all_real_imbalance.py default_all2all_gates_logs default_all2all_real_imbalance.json
```

**Method 2: Aggregated expert assignment (newer method)**
```bash
# For baseline
python new_compute_all_real_imbalance.py baseline_gates_logs baseline_real_imbalance.json

# For defaultAll2All
python new_compute_all_real_imbalance.py default_all2all_gates_logs default_all2all_real_imbalance.json
```

**For a single directory:**
```bash
# Per-step method
python get_real_imbalance.py <gates_logs_directory>

# Aggregated method
python new_real_imbalance.py <gates_logs_directory>
```

#### Generate Plots

Plot performance vs predicted imbalance:

```bash
python plot_metrics_vs_imbalance_score.py \
    --baseline-results baseline_results_json \
    --all2all-results default_all2all_results_json \
    --metric mean_ttft_ms \
    --output graphs/ttft_vs_imbalance_score.png
```

Plot performance vs real imbalance:

```bash
python plot_metrics_vs_real_imbalance.py \
    --baseline-imbalance baseline_real_imbalance.json \
    --baseline-results baseline_results_json \
    --all2all-imbalance default_all2all_real_imbalance.json \
    --all2all-results default_all2all_results_json \
    --metric mean_tpot_ms \
    --output graphs/tpot_vs_real_imbalance.png
```

## Output Structure

- **`results_json/{index}.json`**: Benchmark results per dataset
  - Contains: performance metrics (TTFT, TPOT, throughput), predicted imbalance score and components
- **`baseline_real_imbalance.json`**: Real imbalance scores (CV) per experiment
  - Maps `gates_logs_XX` to CV score
- **`graphs/`**: Generated visualization plots

## Notes

- Server and client must run on the same compute node
- Ensure GPU resources are allocated before starting the server
- Gate logs are collected automatically during inference (requires vLLM instrumentation)
- Each dataset run processes 30 prompts by default (configurable in `client.sh`)

