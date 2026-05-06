#!/usr/bin/env python3
"""Measure PyTorch profiler overhead on a 2-way tensor-parallel LLM trainer."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, schedule


@dataclass
class StepStats:
    rank: int
    mode: str
    step_times_s: list[float]
    measured_step_times_s: list[float]
    mean_step_time_s: float
    median_step_time_s: float
    tokens_per_s: float
    peak_memory_gib: float
    final_loss: float


class TensorParallelSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, tp_size: int) -> None:
        super().__init__()
        if n_heads % tp_size != 0:
            raise ValueError(f"n_heads={n_heads} must be divisible by tp_size={tp_size}")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_local_heads = n_heads // tp_size
        self.head_dim = d_model // n_heads
        self.local_width = self.n_local_heads * self.head_dim
        self.qkv = nn.Linear(d_model, 3 * self.local_width, bias=False)
        self.out = nn.Linear(self.local_width, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_local_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_local_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_local_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.local_width)
        y = self.out(y)
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y


class TensorParallelMLP(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, tp_size: int) -> None:
        super().__init__()
        if ffn_dim % tp_size != 0:
            raise ValueError(f"ffn_dim={ffn_dim} must be divisible by tp_size={tp_size}")
        local_ffn = ffn_dim // tp_size
        self.gate = nn.Linear(d_model, local_ffn, bias=False)
        self.up = nn.Linear(d_model, local_ffn, bias=False)
        self.down = nn.Linear(local_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(F.silu(self.gate(x)) * self.up(x))
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, tp_size: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = TensorParallelSelfAttention(d_model, n_heads, tp_size)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = TensorParallelMLP(d_model, ffn_dim, tp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TensorParallelLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        n_layers: int,
        seq_len: int,
        tp_size: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.empty(seq_len, d_model))
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, ffn_dim, tp_size) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens) + self.pos_embed[: tokens.size(1)]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["none", "kineto", "execution_trace", "both"], default="none")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--ffn-dim", type=int, default=11008)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--profile-wait", type=int, default=0)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-active", type=int, default=3)
    return parser.parse_args()


def init_dist() -> tuple[int, int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def make_profiler(args: argparse.Namespace, rank: int):
    if args.mode not in {"kineto", "both"}:
        return nullcontext()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    def trace_handler(prof) -> None:
        prof.export_chrome_trace(str(args.out_dir / f"kineto_trace_rank{rank}.json"))

    return profile(
        activities=activities,
        schedule=schedule(
            wait=args.profile_wait,
            warmup=args.profile_warmup,
            active=args.profile_active,
            repeat=1,
        ),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_dist()
    if world_size != 2:
        raise RuntimeError(f"this experiment expects tp_size/world_size=2, got {world_size}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = TensorParallelLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        n_layers=args.n_layers,
        seq_len=args.seq_len,
        tp_size=world_size,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    step_times: list[float] = []
    final_loss = float("nan")

    et = None
    if args.mode in {"execution_trace", "both"}:
        from torch.profiler import ExecutionTraceObserver

        et = ExecutionTraceObserver()
        et.register_callback(str(args.out_dir / f"torch_et_rank{rank}.json"))
        et.start()

    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    with make_profiler(args, rank) as prof:
        for step in range(args.steps):
            tokens = torch.randint(
                0,
                args.vocab_size,
                (args.batch_size, args.seq_len + 1),
                device=device,
                generator=generator,
            )
            x = tokens[:, :-1]
            labels = tokens[:, 1:]

            start = time.perf_counter()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), labels.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            step_times.append(elapsed)
            final_loss = float(loss.detach().float().item())
            if prof is not None:
                prof.step()
            if rank == 0:
                print(
                    f"mode={args.mode} step={step + 1}/{args.steps} "
                    f"time_s={elapsed:.4f} loss={final_loss:.4f}",
                    flush=True,
                )

    if et is not None:
        et.stop()
        et.unregister_callback()

    measured = step_times[args.warmup_steps :]
    stats = StepStats(
        rank=rank,
        mode=args.mode,
        step_times_s=step_times,
        measured_step_times_s=measured,
        mean_step_time_s=statistics.fmean(measured),
        median_step_time_s=statistics.median(measured),
        tokens_per_s=(args.batch_size * args.seq_len * world_size) / statistics.fmean(measured),
        peak_memory_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
        final_loss=final_loss,
    )
    with (args.out_dir / f"summary_rank{rank}.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
