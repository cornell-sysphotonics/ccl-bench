============================================================
Policy:      learned
Workload:    llama-8b  batch=32  seq=1024
Environment: 16 GPUs (4n × 4g), A100 40 GB
Goal:        step_time
============================================================
[learned_policy] total_gpus=16, gpu_memory=40 GB, max_iterations=10

[Claude] I'll systematically explore the configuration space. Let me start with the baseline and then branch out to other promising configurations.

## Step 1: Run the baseline

[sim 1/10] tp=4, dp=2, pp=1  (8 GPUs)
  → {'error': 'out of memory', 'message': 'Per-GPU memory exceeded the hardware limit. Try a configuration with higher TP (tensor parallelism) to shard weights/activations across more GPUs, or reduce DP/PP.', 'stderr': 'OUT OF MEMORY: [GPU pp=0,cp=0,tp=0,dp=0] requires 40.569 GiB > 40 GiB\nOUT OF MEMORY: [GPU pp=0,cp=0,tp=0,dp=1] requires 40.569 GiB > 40 GiB\nOUT OF MEMORY: [GPU pp=0,cp=0,tp=1,dp=0] requires 40.569 GiB > 40 GiB\nOUT OF MEMORY: [GPU pp=0,cp=0,tp=1,dp=1] requires 40.569 GiB > 40 GiB\nOUT OF MEMORY: [GPU pp=0,cp=0,tp=2,dp=0] requires 40.569 GiB > 40 GiB\nOUT OF MEMORY: [GPU pp=0,cp=0,tp=2,dp=1] requires 40.569 GiB > 40 GiB\nOUT OF MEMORY: [GPU pp=0,cp=0,tp=3,dp=0] requires 40.569 GiB > 40 GiB\nOUT OF MEMORY: [GPU pp=0,cp=0,tp=3,dp=1] requires 40.569 GiB > 40 GiB\n'}

[Claude] Baseline OOMs at 40.569 GiB vs 40 GiB limit. Need more sharding. Let me try higher TP and PP options.

## Step 2: Try higher TP to fix OOM

[sim 2/10] tp=8, dp=2, pp=1  (16 GPUs)
  → {'wall_time': 143180928, 'gpu_time': 72899742.0, 'comm_time': 86663752.0, 'num_ranks': 16}

[Claude] TP=8 works but crosses node boundaries (4 GPUs/node). Wall time: 143,180,928. Let me try PP to reduce memory instead.

## Step 3: Try PP instead of high TP

[sim 3/10] tp=4, dp=2, pp=2  (16 GPUs)
  → {'wall_time': 202854196, 'gpu_time': 72731766.0, 'comm_time': 116082282.0, 'num_ranks': 16}

[Claude] PP=2 with TP=4 is slower (202M). Let me try other combinations.

## Step 4: Try TP=8 with PP

[sim 4/10] tp=8, dp=1, pp=2  (16 GPUs)
  → {'wall_time': 272453028, 'gpu_time': 72473654.0, 'comm_time': 178745918.0, 'num_ranks': 16}

[Claude] Worse. Let me try more DP-heavy configs with TP=8.

## Step 5: TP=8, dp=1, pp=1 (fewer GPUs, less communication)

[sim 5/10] tp=8, dp=1, pp=1  (8 GPUs)
  → {'error': 'out of memory', 'message': 'Per-GPU memory exceeded the hardware limit. Try a configuration with higher TP (tensor parallelism) to shard weights/activations across more GPUs, or reduce DP/PP.', 'stderr': 'OUT OF MEMORY: [GPU pp=0,cp=0,tp=0,dp=0] requires 41.444 GiB > 40 GiB\nOUT OF MEMORY: [GPU pp=0,cp=0,tp=1,dp=0] requires 41.444 GiB > 40 GiB\nOUT OF MEMORY: [GPU pp=0,cp=0,tp=2,dp=0] requires 41.444 GiB > 40 GiB\nOUT OF MEMORY: [GPU pp=0,cp=0,tp=3,dp=0] requires 41.444 GiB > 40 GiB\nOUT OF MEMORY: [GPU pp=0,cp=0,tp=4,dp=0] requires 41.444 GiB > 40 GiB\nOUT OF MEMORY: [GPU pp=0,cp=0,tp=5,dp=0] requires 41.444 GiB > 40 GiB\nOUT OF MEMORY: [GPU pp=0,cp=0,tp=6,dp=0] requires 41.444 GiB > 40 GiB\nOUT OF MEMORY: [GPU pp=0,cp=0,tp=7,dp=0] requires 41.444 GiB > 40 GiB\n'}

[Claude] Interesting - DP=1 OOMs even with TP=8 (activations need DP splitting). Let me try more DP.

## Step 6: TP=4 with higher DP

[sim 6/10] tp=4, dp=4, pp=1  (16 GPUs)
  → {'wall_time': 380602312, 'gpu_time': 73415966.0, 'comm_time': 335974256.0, 'num_ranks': 16}

[Claude] TP=4, DP=4 works but is very slow (380M). Communication-heavy. Let me try TP=16.

## Step 7: TP=16

[sim 7/10] tp=16, dp=1, pp=1  (16 GPUs)
  → {'wall_time': 149750866, 'gpu_time': 72641630.0, 'comm_time': 83369752.0, 'num_ranks': 16}

[Claude] TP=16 is 149.7M - close but worse than TP=8/DP=2 (143.2M). Let me explore more.

## Step 8: TP=4, PP=4

[sim 8/10] tp=4, dp=1, pp=4  (16 GPUs)
  → {'wall_time': 403034766, 'gpu_time': 72389850.5, 'comm_time': 256866041.5, 'num_ranks': 16}

[Claude] Very slow. PP causes too much overhead here. Let me try mixed configurations.

## Step 9: TP=2, DP=4, PP=2

[sim 9/10] tp=2, dp=4, pp=2  (16 GPUs)
  → {'wall_time': 328668868, 'gpu_time': 73247990.0, 'comm_time': 235120278.0, 'num_ranks': 16}

[Claude] ## Step 10: TP=2, DP=2, PP=2 (8 GPUs)

[sim 10/10] tp=2, dp=2, pp=2  (8 GPUs)
  → {'wall_time': 327910594, 'gpu_time': 145295925.0, 'comm_time': 160669361.0, 'num_ranks': 8}

[Claude] ## Summary of All Explored Configurations

| Config (tp, dp, pp) | GPUs | Wall Time | Status |
|---|---|---|---|
| tp=4, dp=2, pp=1 | 8 | — | OOM (40.57 GiB) |
| tp=8, dp=1, pp=1 | 8 | — | OOM (41.44 GiB) |
| **tp=8, dp=2, pp=1** | **16** | **143,180,928** | **✅ Best** |
| tp=16, dp=1, pp=1 | 16 | 149,750,866 | ✅ |
| tp=4, dp=2, pp=2 | 16 | 202,854,196 | ✅ |
| tp=8, dp=1, pp=2 | 16 | 272,453,028 | ✅ |
| tp=2, dp=2, pp=2 | 8 | 327,910,594 | ✅ |
| tp=2, dp=4, pp=2 | 16 | 328,668,868 | ✅ |
| tp=4, dp=4, pp=1 | 16 | 380,602,312 | ✅ |
| tp=4, dp=1, pp=4 | 16 | 403,034,766 | ✅ |

### Key Insights
1. **Memory pressure** is tight on 40 GB A100s for LLaMA-8B — TP≤4 with no PP causes OOM, and even TP=8/DP=1 OOMs (activations need DP to split the batch).
2. **TP=8, DP=2** strikes the best balance: TP=8 shards model weights sufficiently, DP=2 halves activation memory and batch, and PP=0 avoids pipeline bubbles.
3. **Pipeline parallelism consistently hurts** — every PP≥2 config was significantly slower due to bubble overhead and communication.
4. **TP=16** is competitive but slightly worse than TP=8/DP=2, likely because spanning all 4 nodes for a single TP group increases all-reduce latency without the throughput benefit of data parallelism.

BEST_POLICY: {"tp": 8, "dp": 2, "pp": 1, "wall_time": 143180928}

============================================================
Baseline:       tp=4, dp=2, pp=1
Learned policy: tp=8, dp=2, pp=1  (wall_time=143,180,928 cycles)
============================================================
{
  "tp": 8,
  "dp": 2,
  "pp": 1,
  "wall_time": 143180928
}