# qwen3-4b-vllm-tp2-perlmutter[mscclpp]

Online serving of **Qwen3-4B** on **NVIDIA A100-40GB (Perlmutter)** with **TP=2**, communication library: **NCCL+MSCCL++ (AllGather only)**.

## Files

| File | Purpose |
|---|---|
| `qwen3-4b-vllm-tp2-perlmutter[mscclpp].yaml` | Workload card (read by `tools/main.py` to compute metrics) |
| `server.sh` | Exact server launch command for this variant |
| `client.sh` | Exact client benchmark command (run after server is ready) |
| `*.sqlite` | nsys-exported SQLite trace (the file metric tools actually read) |

## How this trace was produced

A 4× A100 Perlmutter node was allocated, then this exact variant was run by [scripts/run_qwen3_4b_perlmutter_batch.sh](../../scripts/run_qwen3_4b_perlmutter_batch.sh) (a one-shot batch runner that does a warmup + four variants). The end-to-end workflow is in [scripts/qwen3_4b_perlmutter_workflow.md](../../scripts/qwen3_4b_perlmutter_workflow.md).

## Workload params

| Param | Value |
|---|---|
| dataset | random |
| input_len | 1024 |
| output_len | 128 |
| num_prompts | 200 |
| request_rate | 8 RPS |
| TP × DP × PP × CP × EP | 2 × 1 × 1 × 1 × 1 |
| precision | bf16 |
| `--enforce-eager` | yes (no cudagraph) |

## Reproducing

```bash
salloc --nodes 1 --qos interactive --time 00:30:00 --constraint gpu --gpus 4 --account m4999
# (on compute node, in cvllm env)
bash server.sh   # one shell
bash client.sh   # another shell, after /v1/models is up
# Ctrl+C the server when client finishes; nsys finalizes .nsys-rep + .sqlite
```
