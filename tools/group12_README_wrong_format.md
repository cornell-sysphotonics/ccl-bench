# Tools for Metric Extraction and Visualization

This directory contains tools for extracting performance metrics from experiment traces and generating comparative visualizations.

## Overview

The tools directory provides:
- **Metric Extraction**: Extract single numeric metrics from trace directories
- **Comparative Plotting**: Generate plots comparing metrics across multiple experiments
- **Validation**: Automatic checks to ensure trace compatibility with requested metrics

## Available Metrics

### Inference Metrics

These metrics require traces from inference experiments (`phase: inference` in workload YAML):

1. **TTFT** (Time to First Token)
   - **Location**: `TTFT-group_12/`
   - **Extracts**: `mean_ttft_ms` from benchmark JSON files
   - **Plotting**: Available - compares across `max-num-seqs` and `max-num-batched-tokens`

2. **TPOT** (Time Per Output Token)
   - **Location**: `TPOT-group_12/`
   - **Extracts**: `mean_tpot_ms` from benchmark JSON files
   - **Plotting**: Available - compares across `max-num-seqs` and `max-num-batched-tokens`

3. **request_throughput**
   - **Location**: `requestThroughput-group_12/`
   - **Extracts**: `request_throughput` from benchmark JSON files
   - **Plotting**: Available - compares across configurations

4. **output_throughput**
   - **Location**: `outputThroughput-group_12/`
   - **Extracts**: `output_throughput` from benchmark JSON files
   - **Plotting**: Available - compares across configurations

### MoE-Specific Metrics

5. **token_to_expert_assignment**
   - **Location**: `tokenToExpertAssignment-group_12/`
   - **Extracts**: Expert load distribution from gate logs
   - **Output Format**: JSON with totals, normalized distribution, and summary statistics (entropy, CV, max/min ratio)
   - **Requirements**: MoE model (`moe: true` in workload YAML) with `*gates_logs*` directory containing `gates_logs_*/gate_logs_*.json` files
   - **Note**: Gate logs are typically in `scripts/load_imbalance_experiment-group12/baseline_gates_logs/` or `scripts/load_imbalance_experiment-group12/default_all2all_gates_logs/`

6. **communicationOverhead** (EPLB Analysis)
   - **Location**: `communicationOverhead-group_12/`
   - **Analyzes**: CUDA kernel summary CSVs from Nsight Systems traces
   - **Purpose**: Compares Expert Parallel Load Balancing (EPLB) ON vs OFF performance
   - **Input**: Two CSV files exported from Nsight Systems CUDA GPU Kernel Summary
   - **Extracts**: 
     - MoE kernel metrics (fused_moe_kernel stats, timing, variability)
     - NCCL communication metrics (AllToAll, AllReduce timing and instances)
     - Attention kernel timing
     - Load balance metrics (CV, max/min ratios)
   - **Output Format**: JSON file with detailed metrics and comparison results
   - **Requirements**: CSV files exported from Nsight Systems traces

## Prerequisites

### Python Dependencies

```bash
pip install PyYAML matplotlib pandas numpy
```

Required packages:
- `PyYAML` - For parsing workload YAML configuration files
- `matplotlib` - For generating plots (if using plotting scripts)
- `pandas`, `numpy` - For CSV parsing and numerical analysis (communicationOverhead tool)
- Standard library: `json`, `sys`, `pathlib`, `collections`, `math`, `subprocess`, `argparse`, `re`

### Trace Directory Structure

Trace directories must follow this structure:

```
<trace_directory>/
├── *.yaml                    # Workload configuration file (required)
├── results_json/             # Directory containing benchmark results
│   └── *.json               # JSON files with performance metrics
└── [gates_logs/]            # (Optional) For token_to_expert_assignment metric
    └── gates_logs_*/        # Subdirectories with gate log files
        └── gate_logs_*.json # Gate log JSON files
```

### Workload YAML Requirements

The workload YAML file must contain:

- **For inference metrics** (TTFT, TPOT, throughput):
  ```yaml
  workload:
    model:
      phase: inference  # Required
  ```

- **For token_to_expert_assignment**:
  ```yaml
  workload:
    model:
      moe: true  # Required
  ```

- **Results JSON format**:
  Inference metrics expect JSON files in `results_json/` with fields like:
  - `mean_ttft_ms`: Time to first token (milliseconds)
  - `mean_tpot_ms`: Time per output token (milliseconds)
  - `request_throughput`: Requests per second
  - `output_throughput`: Output tokens per second

## Usage

### Unified Interface (Recommended)

Use `main.py` as the unified entry point for all metric extraction:

```bash
python tools/main.py --trace <trace_directory> --metric <metric_name>
```

**Examples**:
```bash
# Extract TTFT from a baseline experiment (note: quote paths with square brackets in zsh)
python tools/main.py --trace "trace_collection/Mixtral8x7B-vllmTP4-Perlmutter[baseline_32_8192]-group12" --metric TTFT

# Extract request throughput
python tools/main.py --trace "trace_collection/Mixtral8x7B-vllmEP4-Perlmutter[defaultall2all_32_8192]-group12" --metric request_throughput

# Extract expert assignment metrics (use load_imbalance_experiment directory with gates_logs)
# Note: If multiple gates_logs directories exist, the script uses the first one alphabetically (typically baseline_gates_logs)
python tools/main.py --trace "scripts/load_imbalance_experiment-group12" --metric token_to_expert_assignment

# To specify a specific gates_logs directory, point directly to it:
python tools/main.py --trace "scripts/load_imbalance_experiment-group12/baseline_gates_logs" --metric token_to_expert_assignment
python tools/main.py --trace "scripts/load_imbalance_experiment-group12/default_all2all_gates_logs" --metric token_to_expert_assignment

# Analyze communication overhead (EPLB comparison)
python tools/main.py --metric communicationOverhead --csv1 tools/communicationOverhead-group_12/EPLBOFF.csv --csv2 tools/communicationOverhead-group_12/EPLBON.csv
```

**Note**: Always quote trace directory paths to avoid zsh parsing square brackets 

**Available metric names**:
- `TTFT`
- `TPOT`
- `request_throughput`
- `output_throughput`
- `token_to_expert_assignment`
- `communicationOverhead` (requires `--csv1` and `--csv2` instead of `--trace`)

**Output**:
- For numeric metrics: Prints a single float value (average across all JSON files)
- For `token_to_expert_assignment`: Prints JSON object with detailed statistics
- For `communicationOverhead`: Prints detailed comparison to stdout and saves JSON results to `kernel_comparison_results.json`

### Direct Script Execution

You can also run extraction scripts directly:

```bash
# Note: Quote paths with square brackets in zsh
python tools/TTFT-group_12/extract.py "<trace_directory>"
python tools/TPOT-group_12/extract.py "<trace_directory>"
python tools/requestThroughput-group_12/extract.py "<trace_directory>"
python tools/outputThroughput-group_12/extract.py "<trace_directory>"
python tools/tokenToExpertAssignment-group_12/extract.py "<trace_directory>"
```

**Examples**:
```bash
# Extract TTFT from trace collection experiment
python tools/TTFT-group_12/extract.py "trace_collection/Mixtral8x7B-vllmTP4-Perlmutter[baseline_32_8192]-group12"

# Extract token_to_expert_assignment from load imbalance experiment
python tools/tokenToExpertAssignment-group_12/extract.py "scripts/load_imbalance_experiment-group12"
# This will automatically find baseline_gates_logs or default_all2all_gates_logs
```

### Communication Overhead Analysis (EPLB)

The `communicationOverhead-group_12` tool analyzes CUDA kernel summaries from Nsight Systems traces to compare EPLB ON vs OFF.

**Via main.py (recommended)**:
```bash
python tools/main.py --metric communicationOverhead --csv1 <eplb_off.csv> --csv2 <eplb_on.csv>
```

**Direct script execution**:
```bash
python tools/communicationOverhead-group_12/analyzeEPLB.py <eplb_off.csv> <eplb_on.csv>
```

**Preparing CSV files**:
1. Open trace in Nsight Systems (`nsys-ui`)
2. Navigate to "CUDA GPU Kernel Summary" in Analysis Summary
3. Right-click → Export to CSV
4. Export separate CSVs for EPLB OFF and EPLB ON traces

**Example**:
```bash
# CSV files are in tools/communicationOverhead-group_12/ directory
python tools/main.py --metric communicationOverhead --csv1 tools/communicationOverhead-group_12/EPLBOFF.csv --csv2 tools/communicationOverhead-group_12/EPLBON.csv

# Or if running from tools/communicationOverhead-group_12/ directory:
cd tools/communicationOverhead-group_12
python ../main.py --metric communicationOverhead --csv1 EPLBOFF.csv --csv2 EPLBON.csv
```

**Output**:
- Prints detailed comparison to stdout (MoE kernel stats, NCCL metrics, load balance improvements)
- Saves JSON results to `kernel_comparison_results.json` in the tool directory

**Metrics analyzed**:
- Fused MoE kernel timing (total, avg, min, max, stddev, CV)
- NCCL communication time (AllToAll, AllReduce)
- Communication/compute ratio
- Load balance metrics (CV reduction, max/min ratio)

### Plotting

Plotting scripts generate comparative visualizations across multiple experiments:

```bash
python tools/<metric>-group_12/plot.py [trace_collection_dir] [output_dir]
```

**Available plot scripts**:
- `TTFT-group_12/plot.py`
- `TPOT-group_12/plot.py`
- `requestThroughput-group_12/plot.py`
- `outputThroughput-group_12/plot.py`

**Arguments**:
- `trace_collection_dir` (optional): Path to trace_collection directory (default: `trace_collection/`)
- `output_dir` (optional): Directory to save plots (default: `graphs/`)

**Examples**:
```bash
# Plot TTFT comparisons
python tools/TTFT-group_12/plot.py trace_collection/ graphs/

# Plot TPOT metrics
python tools/TPOT-group_12/plot.py trace_collection/ graphs/

# Plot output throughput
python tools/outputThroughput-group_12/plot.py trace_collection/ graphs/
```

**Plot Features**:
- Compares Baseline vs. DefaultAll2All implementations
- Two plots per metric:
  - `max-num-seqs` vs. metric (fixing `max-num-batched-tokens = 8192`)
  - `max-num-batched-tokens` vs. metric (fixing `max-num-seqs = 32`)
- Automatically filters out EPLB experiments (`*_eplboff*`, `*_eplbon*`)
- Includes fixed configuration annotations on plots

## Validation

Each extraction script includes validation logic:

1. **Trace directory exists**: Checks that the provided directory exists
2. **Workload YAML present**: Verifies at least one YAML file exists in the trace directory
3. **YAML format valid**: Parses and validates workload YAML structure
4. **Metric compatibility**: 
   - Inference metrics check `phase: inference`
   - `token_to_expert_assignment` checks `moe: true` and `gates_logs` directory
5. **Results directory**: Verifies `results_json/` exists (for inference metrics)

## Output Formats

### Numeric Metrics (TTFT, TPOT, throughput)

Outputs a single float value representing the average across all JSON files in `results_json/`:

```bash
$ python tools/main.py --trace <dir> --metric TTFT
4259.41
```

### token_to_expert_assignment

Outputs a JSON object:

```json
{
  "totals": [1000.0, 980.0, 1050.0, ...],
  "normalized": [0.125, 0.1225, 0.13125, ...],
  "summary": {
    "entropy": 2.95,
    "normalized_entropy": 0.95,
    "coefficient_of_variation": 0.15,
    "max_min_ratio": 1.25,
    "total_tokens": 8000.0,
    "num_experts": 8,
    "total_steps": 100
  }
}
```

## Directory Naming Convention

The plotting scripts expect trace directories to follow the naming convention:

```
Mixtral8x7B-<framework>-Perlmutter[<model>_<max_seqs>_<max_tokens>]-group12
```

Where:
- `framework`: `vllmTP4` (Baseline) or `vllmEP4` (DefaultAll2All)
- `model`: `baseline` or `defaultall2all`
- `max_seqs`: Maximum number of sequences (8, 16, 32, 64)
- `max_tokens`: Maximum batched tokens (2048, 4096, 8192, 16384)

Examples:
- `Mixtral8x7B-vllmTP4-Perlmutter[baseline_32_8192]-group12`
- `Mixtral8x7B-vllmEP4-Perlmutter[defaultall2all_64_16384]-group12`

Directories with `_eplboff` or `_eplbon` in the name are automatically excluded from plots.


