# Hardware-aware A100 versus H100 interpretation

## Scope

This experiment holds the DeepSeek-V3-16B workload and execution plan fixed:
BF16, eight GPUs, two nodes, TP=4, DP-shard=2, EP=8, PP=1, batch size 8,
sequence length 1024, and six steps. It compares complete platforms,
A100/Perlmutter and H100/Eos. It does not isolate the GPU architecture:
the host, intra-node fabric, scale-out network, and exact PyTorch build also
change.

## Hardware ceilings

The Perlmutter baseline uses four 40 GB A100 GPUs and four 200 Gb/s
Slingshot-11 NICs per GPU node. The Eos run's runtime inventory identifies
80 GB HBM3 H100 GPUs; Eos is a DGX H100/Quantum-2 InfiniBand system.

| Hardware quantity | A100/Perlmutter | H100/Eos | H100/A100 |
|---|---:|---:|---:|
| Dense BF16 Tensor Core peak per GPU | 312 TFLOP/s | 989 TFLOP/s | 3.17x |
| Dense BF16 peak across 8 used GPUs | 2.496 PFLOP/s | 7.912 PFLOP/s | 3.17x |
| HBM capacity per GPU | 40 GB HBM2 | 80 GB HBM3 | 2.00x |
| HBM bandwidth per GPU | 1.555 TB/s | 3.35 TB/s | 2.15x |
| NVLink bandwidth per GPU | 600 GB/s | 900 GB/s | 1.50x |
| Network rate per adapter | 200 Gb/s | up to 400 Gb/s | up to 2.00x |
| Installed node injection ceiling | 4×200 = 0.8 Tb/s | 8×400 = 3.2 Tb/s | up to 4.00x |

The node-injection row is a hardware ceiling, not measured application
throughput. The H100 NCCL log enumerates eight InfiniBand HCAs on each node,
but the trace does not establish that every link was saturated.

Sources: [NERSC Perlmutter architecture](https://docs.nersc.gov/systems/perlmutter/architecture/),
[NERSC Perlmutter hardware overview](https://www.nersc.gov/assets/Uploads/02-Intro-To-Perlmutter-Srinivasan.pdf),
[NVIDIA A100 specifications](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf),
[NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/),
[NVIDIA Eos architecture](https://blogs.nvidia.com/blog/eos/), and
[NVIDIA DGX H100 networking](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html).
Dense BF16 ceilings are used because this workload does not enable NVIDIA
2:4 structured sparsity.

## Relating the hardware to the measured metrics

| Observed metric | A100/Perlmutter | H100/Eos | Change |
|---|---:|---:|---:|
| Log-derived step time | 3.67276 s | 0.67166 s | 5.47x lower |
| CCL-Bench MFU | 8.7359% | 15.0664% | 1.72x |
| Effective aggregate BF16 rate implied by MFU | 218 TFLOP/s | 1,192 TFLOP/s | 5.47x |
| Mean SM coverage | 14.3896% | 27.9961% | 1.95x |
| Communication time | 3.08812 s | 0.33838 s | 9.13x lower |
| Communication fraction | 89.8156% | 76.5356% | -13.28 percentage points |
| Compute/communication overlap | 2.7624% | 3.4356% | 1.24x |
| Load-imbalance ratio | 1.02785 | 1.64222 | 1.60x worse |

### Compute contribution

The 5.47x step-time improvement is consistent with both a faster compute
roof and better achieved utilization:

`3.17x BF16 peak ratio × 1.72x MFU ratio = 5.47x effective-FLOP-rate ratio`.

Equivalently, applying CCL-Bench MFU to the eight-GPU hardware roofs gives
about 218 aggregate TFLOP/s on A100 and 1,192 aggregate TFLOP/s on H100.
Mean SM coverage nearly doubles, while dominant-kernel concentration drops
from 42.98% to 30.13%. These metrics indicate greater SM activity and a
less concentrated kernel bottleneck on the H100 platform.

The absolute MFU values should be treated as CCL-Bench values. The current
MoE cards omit active-parameter count, so the metric charges all model
parameters rather than only routed experts. The relative calculation is
still like-for-like.

### Communication contribution

This EP8 workload communicates across all eight ranks, including the node
boundary. The A100 card records NCCL Socket/TCP over Slingshot. In contrast,
the H100 runtime log shows:

- eight InfiniBand HCAs visible to NCCL per node;
- `IBext_v11/GDRDMA` for inter-node channels;
- P2P/CUMEM for intra-node paths;
- 16 NCCL channels; and
- Hopper NVLS multicast availability.

The raw hardware ratios are 1.5x for NVLink, up to 2x per network adapter,
and up to 4x for installed node injection bandwidth. They do not by
themselves explain a 9.13x reduction in measured communication time. The
remainder is consistent with a more efficient transport path: direct
GPU-to-NIC RDMA, more parallel network rails/channels, NVSwitch/NVLS
collective support, and reduced software/host overhead relative to the
A100 run's Socket/TCP path. The traces establish correlation, not a
controlled attribution among those mechanisms.

Communication still occupies 76.54% of the H100 trace. Therefore the
result does not imply that Hopper removed the communication bottleneck;
it shows that the bottleneck became much smaller in absolute time while
remaining the largest optimization target.

### Memory and imbalance

H100 provides a 2.15x higher theoretical HBM bandwidth ceiling, but the
captured H100 Kineto window has no memory-copy events with byte counts.
CCL-Bench therefore reports H100 memory bandwidth as `N/A`. The experiment
does not support claiming that HBM bandwidth caused a measured speedup.

The load-imbalance ratio worsens from 1.03 to 1.64. With faster kernels and
communication, expert-routing skew and rank-level jitter consume a larger
fraction of a shorter step. This is a plausible explanation, not a proven
hardware effect, and motivates collecting more profiled steps.

## Paper-ready interpretation

For the matched EP8 DeepSeek-V3-16B workload, the H100/Eos run reduces
step time from 3.673 s to 0.672 s (5.47x). The result is quantitatively
consistent with a 3.17x increase in dense BF16 peak throughput and a 1.72x
increase in CCL-Bench MFU, which together imply a 5.47x higher effective
compute rate. The larger system-level gain comes from communication:
measured communication time falls 9.13x, while the H100 platform raises
NVLink bandwidth from 600 to 900 GB/s, doubles per-adapter network rate
from 200 to up to 400 Gb/s, and uses NCCL GDRDMA across multiple
InfiniBand rails rather than the baseline's Socket/TCP path. Because the
host and network change with the GPU platform, these data demonstrate that
CCL-Bench captures and explains performance across heterogeneous hardware
stacks; they should not be framed as a GPU-only Hopper-versus-Ampere
microarchitecture result.
