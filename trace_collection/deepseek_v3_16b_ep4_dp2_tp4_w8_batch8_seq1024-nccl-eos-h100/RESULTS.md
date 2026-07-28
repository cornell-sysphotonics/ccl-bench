# H100 rerun result

The matched DeepSeek-V3-16B training run completed successfully on two Eos
nodes. Slurm job `5716497` exited `0:0` after `00:04:51`; its enforced time
limit was `01:00:00`.

## Configuration held fixed

- TorchTitan commit `4a57715f34338416d467dfc5fb54a9b69c938db5`
- BF16, NCCL, two nodes, four GPUs used per node
- TP=4, DP-shard=2, EP=4, PP=1
- Global batch size 8, sequence length 1024, six training steps
- C4 test data, AdamW, learning rate 8e-4

The H100 run used the date-matched NVIDIA PyTorch 26.04 container
(`2.12.0a0+0291f960b6.nv26.04.48445190`), CUDA 13.2, and NCCL 2.29.7.
The A100 benchmark row records PyTorch
`2.12.0.dev20260408+cu128` and NCCL 2.29.7.

## Representative CCL-Bench metrics

| Metric | A100/Perlmutter | H100/Eos | H100/A100 |
|---|---:|---:|---:|
| Average step time (s) | 11.9832 | 0.76579 | 0.06391 |
| CCL-Bench MFU (%) | 2.6775 | 13.2079 | 4.9329 |
| Mean SM coverage (%) | 10.3779 | 21.0440 | 2.0278 |
| Communication time (s) | 11.6966 | 0.61091 | 0.05223 |
| Communication fraction (%) | 95.2327 | 82.9793 | 0.87133 |
| Compute/communication overlap (%) | 4.5587 | 9.9243 | 2.1770 |
| Memory bandwidth (GB/s) | 422.685 | 311.836 | 0.73775 |
| Load-imbalance ratio | 1.01685 | 1.58105 | 1.55485 |

For this profiled step, the H100/Eos platform had 15.65x lower step time,
19.15x lower communication time, 2.03x mean SM coverage, and 2.18x
compute/communication overlap. The metrics also expose regressions rather
than only favorable results: sampled memory bandwidth was 26.2% lower and
the load-imbalance ratio was 55.5% higher.

## Interpretation and provenance

This is a like-for-like workload and execution-plan comparison, not a
GPU-only causal comparison. Moving from Perlmutter to Eos also changes the
host and scale-out network: the A100 row used Slingshot/Socket TCP, while
the H100 run used Eos InfiniBand with NCCL GDRDMA.

CCL-Bench MFU is reported for consistency with the existing A100 row. Its
MoE formula uses total parameters because the benchmark cards do not record
an active-parameter count, so its absolute value differs from TorchTitan's
active-parameter MFU. The cross-platform CCL-Bench ratio is computed with
the same definition on both rows.

The reported H100 step time is derived from TorchTitan's step-4 TPS log:
`(local_batch_size * seq_len) / (TP * tps)`, averaged across all eight ranks.
The result is 0.765787 seconds. The corresponding Kineto `ProfilerStep#3`
ranges report 0.766348 seconds, a 0.07% difference caused primarily by the
log rounding TPS to an integer.

The H100 result contains eight Kineto JSON traces, one per rank, totaling
366.6 MB. The trace captures the same single training step; this is
sufficient to exercise the metric pipeline but should not be presented as
a low-variance performance study.

Artifacts:

- `/lustre/fsw/nvr_nrg_misc/erding/ccl-bench/deepseek_v3_16b_ep4_dp2_tp4_w8_batch8_seq1024-nccl-eos-h100-5716497`
- `a100-vs-h100-metrics.md` and `a100-vs-h100-metrics.json`
- `profile_traces/iteration_4/rank{0..7}_trace.json`
- `logs/node-rank-{0,1}.log`
- `environment-rank{0,1}.txt`
