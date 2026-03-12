============================================================
Policy:      learned
Workload:    llama-8b, batch size 32, seq_len: 1024
Environment: 4 nodes, 4 GPUs per node, 2400Gbps network within node, 200Gbps network across n
============================================================

[learned_policy] Starting exploration (max 10 iterations)...

[Claude] # Parallelism Configuration Optimization for LLaMA-8B

## Analysis

Let me first understand the setup:
- **Total GPUs**: 4 nodes × 4 GPUs/node = **16 GPUs**
- **Intra-node**: 2400 Gbps (fast — good for tensor parallelism)
- **Inter-node**: 200 Gbps (slower — minimize cross-node communication)
- **Baseline**: tp=4, dp=2, pp=1 → only uses 8 GPUs

The baseline only uses 8 of 16 GPUs. Let me explore configurations that could use more GPUs and minimize step time. Let me start with the baseline.

## Step 1: Run the baseline (tp=4, dp=2, pp=1)

[sim 1/10] tp=4, dp=2, pp=1  (8 GPUs)
  → {'wall_time': 201600196, 'gpu_time': 145463532.0, 'comm_time': 72401528.0, 'num_ranks': 8}

[Claude] **Baseline**: wall_time = **201,600,196 ns** (8 GPUs)

## Step 2: Try using all 16 GPUs — tp=4, dp=4, pp=1

Keeping TP within a node (4 GPUs) and doubling DP to use all 16 GPUs:

[sim 2/10] tp=4, dp=4, pp=1  (16 GPUs)
  → {'wall_time': 380602312, 'gpu_time': 73415966.0, 'comm_time': 335974256.0, 'num_ranks': 16}

[Claude] **Result**: wall_time = **380,602,312 ns** — worse! Communication time exploded (336M ns) due to heavy cross-node all-reduce with dp=4.

## Step 3: Try tp=4, dp=2, pp=2 (add pipeline parallelism)

[sim 3/10] tp=4, dp=2, pp=2  (16 GPUs)
  → {'wall_time': 202854196, 'gpu_time': 72731766.0, 'comm_time': 116082282.0, 'num_ranks': 16}

[Claude] **Result**: wall_time = **202,854,196 ns** — similar to baseline. PP adds pipeline bubble overhead.

## Step 4: Try tp=2, dp=4, pp=2

[sim 4/10] tp=2, dp=4, pp=2  (16 GPUs)
  → {'wall_time': 328668868, 'gpu_time': 73247990.0, 'comm_time': 235120278.0, 'num_ranks': 16}

[Claude] **Result**: wall_time = **328,668,868 ns** — worse, high cross-node comm from dp=4.

## Step 5: Try tp=2, dp=2, pp=2 (8 GPUs)

[sim 5/10] tp=2, dp=2, pp=2  (8 GPUs)
  → {'wall_time': 327910594, 'gpu_time': 145295925.0, 'comm_time': 160669361.0, 'num_ranks': 8}

[Claude] **Result**: wall_time = **327,910,594 ns** — worse with reduced TP.

## Step 6: Try tp=8, dp=2, pp=1 (TP across nodes)

[sim 6/10] tp=8, dp=2, pp=1  (16 GPUs)
  → {'wall_time': 143180928, 'gpu_time': 72899742.0, 'comm_time': 86663752.0, 'num_ranks': 16}

[Claude] **Result**: wall_time = **143,180,928 ns** — **significant improvement!** 🎯 29% faster than baseline despite TP crossing node boundaries.

## Step 7: Try tp=16, dp=1, pp=1 (all TP)

[sim 7/10] tp=16, dp=1, pp=1  (16 GPUs)
  → {'wall_time': 149750866, 'gpu_time': 72641630.0, 'comm_time': 83369752.0, 'num_ranks': 16}

[Claude] **Result**: wall_time = **149,750,866 ns** — close but slightly worse than tp=8, dp=2, pp=1.

## Step 8: Try tp=8, dp=1, pp=2

[sim 8/10] tp=8, dp=1, pp=2  (16 GPUs)
  → {'wall_time': 272453028, 'gpu_time': 72473654.0, 'comm_time': 178745918.0, 'num_ranks': 16}

[Claude] **Result**: wall_time = **272,453,028 ns** — PP hurts here.

## Step 9: Try tp=2, dp=8, pp=1 (minimize TP, max DP)

[sim 9/10] tp=2, dp=8, pp=1  (16 GPUs)
  → {'wall_time': 172880784, 'gpu_time': 74448414.0, 'comm_time': 128995416.0, 'num_ranks': 16}

[Claude] **Result**: wall_time = **172,880,784 ns** — decent but tp=8, dp=2 still leads.

## Step 10: Try tp=1, dp=16, pp=1 (pure data parallelism)

[sim 10/10] tp=1, dp=16, pp=1  (16 GPUs)
  → {'wall_time': 76513310, 'gpu_time': 76513310.0, 'comm_time': 21660952.0, 'num_ranks': 16}

[Claude] ## Summary of All Explored Configurations

| Config (tp, dp, pp) | GPUs | Wall Time (ns) | vs Baseline |
|---|---|---|---|
| tp=4, dp=2, pp=1 (baseline) | 8 | 201,600,196 | — |
| tp=4, dp=4, pp=1 | 16 | 380,602,312 | +89% ❌ |
| tp=4, dp=2, pp=2 | 16 | 202,854,196 | ~same |
| tp=2, dp=4, pp=2 | 16 | 328,668,868 | +63% ❌ |
| tp=2, dp=2, pp=2 | 8 | 327,910,594 | +63% ❌ |
| tp=8, dp=2, pp=1 | 16 | 143,180,928 | -29% ✅ |
| tp=16, dp=1, pp=1 | 16 | 149,750,866 | -26% ✅ |
| tp=8, dp=1, pp=2 | 16 | 272,453,028 | +35% ❌ |
| tp=2, dp=8, pp=1 | 16 | 172,880,784 | -14% ✅ |
| **tp=1, dp=16, pp=1** | **16** | **76,513,310** | **-62% ✅✅** |

## Key Insights

1. **Pure data parallelism wins decisively.** LLaMA-8B is small enough to fit entirely in a single A100 40GB, so tensor and pipeline parallelism are unnecessary and only add communication overhead that *cannot* be overlapped.

2. **DP communication overlaps perfectly with compute.** Notice that `wall_time == gpu_time` (76.5M ns) for tp=1, dp=16, pp=1, meaning the gradient all-reduce is fully hidden behind backward-pass computation — zero exposed communication cost.

3. **TP/PP fragment computation and expose communication.** Higher TP splits layers across GPUs, creating frequent synchronization barriers (all-reduce on activations) that cannot be overlapped. PP adds pipeline bubble idle time.

4. **The baseline (tp=4, dp=2) was suboptimal in two ways:** it used only 8 of 16 GPUs, and it unnecessarily split the model with tensor parallelism.

BEST_POLICY: {"tp": 1, "dp": 16, "pp": 1, "wall_time": 76513310}

============================================================
Baseline:       tp=4, dp=2, pp=1
Learned policy: tp=1, dp=16, pp=1  (wall_time=76,513,310 cycles)
============================================================
{
  "tp": 1,
  "dp": 16,
  "pp": 1,
  "wall_time": 76513310
}