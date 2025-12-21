# vLLM Profiling on Perlmutter - Quick Start Guide

This guide helps you run the vLLM parallelism profiling experiments on Perlmutter.

## 🚀 Setup (One-Time)

### 1. On Perlmutter

```bash
# Login to Perlmutter
ssh perlmutter.nersc.gov

# Navigate to your workspace
cd $SCRATCH  # or your preferred directory
git clone <your-repo-url> ccl-bench
cd ccl-bench

# Create conda environment
module load python
conda create --name vllm-profiling python=3.10
conda activate vllm-profiling

# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x scripts/*.sh
```

### 2. Download Models (Optional - will auto-download)

```bash
# Login to Hugging Face (if models are gated)
huggingface-cli login

# Pre-download models to avoid timeout during profiling
python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
"
```

## 📋 Running Experiments

### Step 1: Generate All Experiment Configs

```bash
python experiments/generate_all_configs.py
```

This creates three experiment configurations (E1.1–E1.3) in `experiments/configs/`.

### Step 2: Update SLURM Scripts

Edit SLURM scripts to add your allocation:

```bash
# Edit each script in experiments/slurm_scripts/
# Replace: #SBATCH -A <your_allocation>
# With:    #SBATCH -A your_actual_allocation_name
```

### Step 3: Run Baseline Experiment

```bash
# Create logs directory
mkdir -p logs

# Submit baseline job
sbatch experiments/slurm_scripts/run_E1.1.sh

# Check job status
squeue -u $USER

# Monitor output
tail -f logs/E1.1_*.out
```

### Step 4: Run All Single-Parallelism Experiments

```bash
# Run TP experiments
sbatch experiments/slurm_scripts/run_E1.1.sh  # Baseline
sbatch experiments/slurm_scripts/run_E1.2.sh  # TP=2
sbatch experiments/slurm_scripts/run_E1.3.sh  # TP=4

# Check results
ls -lh trace_collection/
```

### Step 5: Generate Workload Cards

```bash
# For each completed experiment
python experiments/generate_workload_card.py \
    --config experiments/configs/E1.1_llama8b_baseline.yaml \
    --output-dir trace_collection/llama-8b-tp1
```

### Step 6: Calculate Metrics

```bash
# Throughput
./scripts/get_throughput_tokens_sec.sh llama-8b-tp1

# Iteration time
./scripts/get_iteration_wall_clock.sh llama-8b-tp1

# Communication overhead
./scripts/get_comm_overhead.sh llama-8b-tp2

# TTFT / TPOT
./scripts/get_ttft.sh llama-8b-tp1
./scripts/get_tpot.sh llama-8b-tp1
```

### Step 7: Analyze Results

```bash
# After running multiple experiments
python experiments/analyze_results.py

# View results
cat experiments/results_summary.csv
open experiments/scaling_analysis.png  # or download to view
```

## 🔬 Advanced: Nsys Profiling

For detailed GPU profiling with Nsys:

```bash
# Run with Nsys
sbatch --export=CONFIG=E1.3_llama8b_tp4.yaml \
    experiments/slurm_scripts/run_with_nsys.sh

# Analyze Nsys report (on Perlmutter or download)
module load nsight-systems
nsys stats trace_collection/llama-8b-tp4/nsys_*.nsys-rep
```

## 📊 Expected Output Structure

After running experiments:

```
trace_collection/
├── llama-8b-tp1/
│   ├── workload_card.yaml
│   ├── torch_et_0.json
│   ├── kineto_trace_0.json
│   ├── timing_stats_0.json
│   └── nsys_0.nsys-rep (if using Nsys)
├── llama-8b-tp2/
│   ├── workload_card.yaml
│   ├── torch_et_0.json
│   ├── torch_et_1.json
│   ├── kineto_trace_0.json
│   ├── kineto_trace_1.json
│   └── ...
└── ...

experiments/
├── results_summary.csv          # Metrics table
└── scaling_analysis.png         # Scaling plots
```

## 🎯 Experiment Checklist

**TP Scaling Experiments**
- [ ] E1.1 - Llama-8B Baseline (1 GPU)
- [ ] E1.2 - Llama-8B TP=2
- [ ] E1.3 - Llama-8B TP=4

## ⚙️ Customization

### Change Sequence Length

Edit experiment config:

```yaml
# experiments/configs/E1.1_llama8b_baseline.yaml
data:
  seq_len: 4096  # Change from 2048
```

### Change Number of Profiling Iterations

```yaml
warmup_iterations: 5    # More warmup
profile_iterations: 10  # More profiling samples
```

### Add Custom Metrics

1. Create new tool in `tools/your_metric/your_metric.py`
2. Update `tools/main.py` to register the metric
3. Create script in `scripts/get_your_metric.sh`

Example:
```python
# tools/sm_utilization/sm_utilization.py
def metric_cal(directory: str) -> float:
    # Parse Nsys or Kineto trace
    # Calculate SM utilization %
    return sm_util_pct
```

## 🐛 Common Issues

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size or sequence length

```yaml
data:
  batch_size: 2    # Reduce
  seq_len: 4096    # Reduce
```

### Issue: "Model not found"

**Solution**: Pre-download or check HuggingFace access

```bash
huggingface-cli login
python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-3.1-8B')"
```

### Issue: "vLLM import error"

**Solution**: Reinstall vLLM

```bash
pip uninstall vllm -y
pip install vllm --no-cache-dir
```

### Issue: "NCCL timeout"

**Solution**: Increase timeout in config or check network

```bash
export NCCL_TIMEOUT=1800  # 30 minutes
```

## 📈 Expected Metrics

Based on Perlmutter A100 GPUs:

| Experiment | GPUs | TP | Expected Throughput | Expected Iter Time |
|------------|------|----|--------------------|-------------------|
| E1.1       | 1    | 1  | ~5K tokens/sec     | ~6s               |
| E1.2       | 2    | 2  | ~8K tokens/sec     | ~4s               |
| E1.3       | 4    | 4  | ~12K tokens/sec    | ~2.7s             |

*Note: Actual numbers will vary based on model, batch size, and sequence length*

## 📚 Next Steps

1. Run all experiments in Phase 1
2. Generate workload cards for each
3. Calculate metrics using provided tools
4. Run analysis script to generate summary
5. Create visualizations
6. Write up findings in project report

## 🔗 Resources

- [Experiments README](experiments/README.md) - Detailed experiment documentation
- [Tools README](tools/README.md) - Metric calculation guide
- [Trace Gen README](trace_gen/README.md) - Trace collection methods
- [vLLM Docs](https://docs.vllm.ai/)
- [Perlmutter Guide](https://docs.nersc.gov/systems/perlmutter/)
