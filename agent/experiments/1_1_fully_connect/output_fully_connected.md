============================================================
Policy:      learned
Workload:    1. llama-8b, batch size 32, seq_len: 1024
Environment: 16 GPUs in total, 800Gbps uniform
============================================================

[learned_policy] Starting exploration (max 10 iterations)...

[Claude] # Parallelism Configuration Optimization for LLaMA-8B

## Analysis

Let me first understand the setup:
- **Total GPUs**: 4 nodes × 4 GPUs/node = **16 GPUs**
- **Baseline**: tp=4, dp=2, pp=1 → uses only 8 GPUs

The baseline only uses 8 of 16 GPUs. Let me start by running it, then explore configurations that better utilize all 16 GPUs.

## Step 1: Run the baseline (tp=4, dp=2, pp=1)

[sim 1/10] tp=4, dp=2, pp=1  (8 GPUs)
  → {'wall_time': 229442156, 'gpu_time': 90499310.0, 'comm_time': 172160344.0, 'num_ranks': 8}

[Claude] **Baseline result**: wall_time = **229,442,156 ns** (8 GPUs). Heavy communication overhead (comm_time >> gpu_time).

## Step 2: Try using all 16 GPUs with more data parallelism

[sim 2/10] tp=4, dp=4, pp=1  (16 GPUs)
  → {'wall_time': 125663468, 'gpu_time': 45568286.0, 'comm_time': 99763088.0, 'num_ranks': 16}

[Claude] **Result**: wall_time = **125,663,468 ns** — big improvement using all 16 GPUs! Let me explore more configs.

## Step 3: Try tp=4, dp=2, pp=2

[sim 3/10] tp=4, dp=2, pp=2  (16 GPUs)
  → {'wall_time': 229755656, 'gpu_time': 45249655.0, 'comm_time': 170144339.5, 'num_ranks': 16}

[Claude] **Result**: wall_time = **229,755,656 ns** — PP=2 doesn't help here. Pipeline overhead is significant.

## Step 4: Try tp=2, dp=8, pp=1

[sim 4/10] tp=2, dp=8, pp=1  (16 GPUs)
  → {'wall_time': 95069548, 'gpu_time': 46049054.0, 'comm_time': 62153744.0, 'num_ranks': 16}

[Claude] **Result**: wall_time = **95,069,548 ns** — excellent! Reducing TP to 2 cuts communication significantly.

## Step 5: Try tp=1, dp=16, pp=1

[sim 5/10] tp=1, dp=16, pp=1  (16 GPUs)
  → {'wall_time': 82693515, 'gpu_time': 47010590.0, 'comm_time': 65829400.0, 'num_ranks': 16}

[Claude] **Result**: wall_time = **82,693,515 ns** — pure DP is even better! Let me verify with a few more configs.

## Step 6: Try tp=2, dp=4, pp=2

[sim 6/10] tp=2, dp=4, pp=2  (16 GPUs)
  → {'wall_time': 149191688, 'gpu_time': 45490039.0, 'comm_time': 96726035.5, 'num_ranks': 16}

[Claude] **Result**: wall_time = **149,191,688 ns** — PP continues to hurt.

## Step 7: Try tp=1, dp=8, pp=2

[sim 7/10] tp=1, dp=8, pp=2  (16 GPUs)
  → {'wall_time': 96870020, 'gpu_time': 45970807.0, 'comm_time': 70033963.5, 'num_ranks': 16}

[Claude] **Result**: wall_time = **96,870,020 ns** — PP adds overhead vs pure DP.

## Step 8: Let me try a couple more edge cases

[sim 8/10] tp=1, dp=4, pp=4  (16 GPUs)
  → {'wall_time': 187677795, 'gpu_time': 45450938.0, 'comm_time': 117960829.0, 'num_ranks': 16}

[Claude] **Result**: wall_time = **187,677,795 ns** — PP=4 is very costly.

## Summary of Results

| Config (tp, dp, pp) | GPUs | Wall Time (ns) | Speedup vs Baseline |
|---------------------|------|----------------|-------------------|
| tp=4, dp=2, pp=1 | 8 | 229,442,156 | 1.00× (baseline) |
| tp=4, dp=4, pp=1 | 16 | 125,663,468 | 1.83× |
| tp=4, dp=2, pp=2 | 16 | 229,755,656 | 1.00× |
| **tp=2, dp=8, pp=1** | 16 | 95,069,548 | 2.41× |
| **tp=1, dp=16, pp=1** | **16** | **82,693,515** | **2.78×** |
| tp=2, dp=4, pp=2 | 16 | 149,191,688 | 1.54× |
| tp=1, dp=8, pp=2 | 16 | 96,870,020 | 2.37× |
| tp=1, dp=4, pp=4 | 16 | 187,677,795 | 1.22× |

## Key Insights

1. **Pipeline parallelism hurts**: For LLaMA-8B, PP introduces pipeline bubbles and adds cross-node point-to-point communication that outweighs any benefits.

2. **Less TP is better**: LLaMA-8B (≈16GB in fp16) fits on a single A100 40GB. TP communication (all-reduce per layer) sits on the critical path and **cannot** be overlapped with compute, so minimizing TP is crucial.

3. **Maximize DP**: Data parallelism gradient all-reduce can be **overlapped** with backward computation, and scaling from dp=2→16 splits the batch (32/16=2 per GPU), dramatically reducing per-GPU compute.

4. **Pure DP wins**: With tp=1, dp=16, pp=1, we get a **2.78× speedup** over baseline by eliminating TP overhead entirely and leveraging compute-communication overlap in DP.

## Recommendation

BEST_POLICY: {"tp": 1, "dp": 16, "pp": 1, "wall_time": 82693515}

============================================================
Baseline:       tp=4, dp=2, pp=1
Learned policy: tp=1, dp=16, pp=1  (wall_time=82,693,515 cycles)
============================================================
{
  "tp": 1,
  "dp": 16,
  "pp": 1,
  "wall_time": 82693515
}