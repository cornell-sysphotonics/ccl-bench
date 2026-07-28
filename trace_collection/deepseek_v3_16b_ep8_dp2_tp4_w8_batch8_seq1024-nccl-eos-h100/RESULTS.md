# H100 EP8 rerun result

The matched DeepSeek-V3-16B EP8 training run completed successfully on two
Eos nodes. Slurm job `5716629` exited `0:0` after `00:04:43`; its enforced
time limit was `01:00:00`.

## Configuration held fixed

- TorchTitan commit `4a57715f34338416d467dfc5fb54a9b69c938db5`
- BF16, NCCL, two nodes, four GPUs used per node
- TP=4, DP-shard=2, EP=8, PP=1
- Global batch size 8, sequence length 1024, six training steps
- C4 test data, AdamW, learning rate 8e-4

The H100 run used the NVIDIA PyTorch 26.04 container
(`2.12.0a0+0291f960b6.nv26.04.48445190`), CUDA 13.2, and NCCL 2.29.7.
The A100 benchmark row records PyTorch
`2.12.0.dev20260408+cu128` and NCCL 2.29.7.

## Representative CCL-Bench metrics

| Metric | A100/Perlmutter | H100/Eos | H100/A100 |
|---|---:|---:|---:|
| Average step time (s) | 3.67276 | 0.67166 | 0.18287 |
| CCL-Bench MFU (%) | 8.7359 | 15.0664 | 1.7247 |
| Mean SM coverage (%) | 14.3896 | 27.9961 | 1.9456 |
| Dominant-kernel concentration (%) | 42.9780 | 30.1287 | 0.7010 |
| Communication time (s) | 3.08812 | 0.33838 | 0.10957 |
| Communication fraction (%) | 89.8156 | 76.5356 | 0.85214 |
| Compute/communication overlap (%) | 2.76243 | 3.43562 | 1.2437 |
| Memory-transfer overhead (%) | 0.7877 | 1.5421 | 1.9577 |
| Load-imbalance ratio | 1.02785 | 1.64222 | 1.5977 |

For the captured step, H100/Eos had 5.47x lower step time, 9.13x lower
communication time, 1.95x mean SM coverage, and 1.24x
compute/communication overlap. Communication fraction fell by 13.28
percentage points. The metrics also expose regressions: memory-transfer
overhead was 1.96x higher and load imbalance was 1.60x higher.

H100 average memory bandwidth is unavailable for this trace because the
captured Kineto window has no memory-copy events carrying byte counts. It
is reported as `N/A`, not zero.

## Interpretation and provenance

This is a like-for-like workload and execution-plan comparison, not a
GPU-only causal comparison. Moving from Perlmutter to Eos also changes the
host and scale-out network: the A100 row used Slingshot/Socket TCP, while
the H100 run used Eos InfiniBand with NCCL GDRDMA.

The reported H100 step time comes from TorchTitan's step-4 TPS log:
`(local_batch_size * seq_len) / (TP * tps)`, averaged across all eight
ranks. CCL-Bench MFU uses total MoE parameters because the cards do not
record active parameters, so its absolute value differs from TorchTitan's
active-parameter MFU; both benchmark rows use the same CCL-Bench
definition.

The H100 result contains eight Kineto JSON traces, one per rank, totaling
351.3 MB. The trace captures one training step, which exercises the metric
pipeline but should not be presented as a low-variance performance study.

A hardware-ceiling analysis and paper-ready interpretation are provided in
`HARDWARE_ANALYSIS.md`.

Artifacts:

- `/lustre/fsw/nvr_nrg_misc/erding/ccl-bench/deepseek_v3_16b_ep8_dp2_tp4_w8_batch8_seq1024-nccl-eos-h100-5716629`
- `a100-vs-h100-metrics.md` and `a100-vs-h100-metrics.json`
- `profile_traces/iteration_4/rank{0..7}_trace.json`
- `logs/node-rank-{0,1}.log`
- `environment-rank{0,1}.txt`
