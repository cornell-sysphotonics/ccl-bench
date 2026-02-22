#!/usr/bin/env python3
"""
Profile AllReduce, AllGather, ReduceScatter on multi-node Slurm systems.
Supports both torch.distributed (nccl/gloo) and torchcomms (ncclx).
Tested for Perlmutter (NERSC).
"""

import os
import time
import argparse
import torch
import torch.distributed as dist
from datetime import datetime, timedelta
from typing import List

# Try to import torchcomms for ncclx support
try:
    from torchcomms import new_comm, ReduceOp
    TORCHCOMMS_AVAILABLE = True
except ImportError:
    TORCHCOMMS_AVAILABLE = False

# -----------------------------
# Slurm → torch.distributed FIX
# -----------------------------
def setup_slurm_env():
    """Setup PyTorch distributed env vars from SLURM (if not already set)"""
    if "SLURM_PROCID" not in os.environ:
        return  # not running under Slurm

    # Set RANK, WORLD_SIZE, LOCAL_RANK if not already set
    os.environ.setdefault("RANK", os.environ["SLURM_PROCID"])
    os.environ.setdefault("WORLD_SIZE", os.environ["SLURM_NTASKS"])
    os.environ.setdefault("LOCAL_RANK", os.environ["SLURM_LOCALID"])

    # Set MASTER_ADDR from SLURM if not already set
    if "MASTER_ADDR" not in os.environ:
        master_addr = None
        
        # Method 1: Try SLURM_STEP_NODELIST (available in srun context)
        if "SLURM_STEP_NODELIST" in os.environ:
            try:
                import subprocess
                result = subprocess.run(
                    ["scontrol", "show", "hostnames", os.environ["SLURM_STEP_NODELIST"]],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                hostnames = result.stdout.strip().split('\n')
                if hostnames and hostnames[0]:
                    master_addr = hostnames[0]
            except:
                pass
        
        # Method 2: Try SLURM_JOB_NODELIST
        if not master_addr and "SLURM_JOB_NODELIST" in os.environ:
            try:
                import subprocess
                result = subprocess.run(
                    ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                hostnames = result.stdout.strip().split('\n')
                if hostnames and hostnames[0]:
                    master_addr = hostnames[0]
            except:
                pass
        
        # Method 3: Use scontrol show job
        if not master_addr and "SLURM_JOB_ID" in os.environ:
            try:
                import subprocess
                result = subprocess.run(
                    ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                for line in result.stdout.split('\n'):
                    if 'NodeList=' in line:
                        nodelist = line.split('NodeList=')[1].split()[0]
                        result2 = subprocess.run(
                            ["scontrol", "show", "hostnames", nodelist],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        hostnames = result2.stdout.strip().split('\n')
                        if hostnames and hostnames[0]:
                            master_addr = hostnames[0]
                            break
            except:
                pass
        
        # Method 4: Fallback to SLURMD_NODENAME (rank 0 should be on first node)
        if not master_addr and "SLURMD_NODENAME" in os.environ and os.environ.get("SLURM_PROCID") == "0":
            master_addr = os.environ["SLURMD_NODENAME"]
        
        if master_addr:
            os.environ["MASTER_ADDR"] = master_addr

    os.environ.setdefault("MASTER_PORT", "29500")


# -----------------------------
# Profiler
# -----------------------------
class CollectiveProfiler:
    def __init__(self, backend: str, output_file: str):
        setup_slurm_env()

        self.backend = backend
        self.output_file = output_file
        self.use_torchcomms = TORCHCOMMS_AVAILABLE and backend == "ncclx"

        # Get rank info from environment
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Set device
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        if self.rank == 0:
            print(f"[DEBUG] RANK={self.rank}, WORLD_SIZE={self.world_size}, LOCAL_RANK={self.local_rank}")
            print(f"[DEBUG] MASTER_ADDR={os.environ.get('MASTER_ADDR', 'NOT SET')}")
            print(f"[DEBUG] MASTER_PORT={os.environ.get('MASTER_PORT', 'NOT SET')}")
            print(f"[DEBUG] Using {'torchcomms' if self.use_torchcomms else 'torch.distributed'}")

        # Initialize process group
        if self.use_torchcomms:
            # Use torchcomms for ncclx
            self.comm = new_comm(backend, self.device, name="profiler_comm")
        else:
            # Use torch.distributed for nccl/gloo
            dist.init_process_group(
                backend=self.backend,
                init_method="env://",
                timeout=timedelta(minutes=10),
            )
            dist.barrier()

        if self.rank == 0:
            print(f"[INIT] Successfully initialized: world_size={self.world_size}, backend={self.backend}\n")

    def warmup(self, tensor, iters=10):
        for _ in range(iters):
            if self.use_torchcomms:
                self.comm.all_reduce(tensor, ReduceOp.SUM, async_op=False)
            else:
                dist.all_reduce(tensor)
        torch.cuda.synchronize()

    def profile_all_reduce(self, numel, iters=50):
        t = torch.randn(numel, device=self.device)
        self.warmup(t)

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iters):
            if self.use_torchcomms:
                self.comm.all_reduce(t, ReduceOp.SUM, async_op=False)
            else:
                dist.all_reduce(t)

        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / iters

    def profile_all_gather(self, numel, iters=50):
        inp = torch.randn(numel, device=self.device)
        out = [torch.empty_like(inp) for _ in range(self.world_size)]

        for _ in range(10):
            if self.use_torchcomms:
                self.comm.all_gather(out, inp, async_op=False)
            else:
                dist.all_gather(out, inp)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            if self.use_torchcomms:
                self.comm.all_gather(out, inp, async_op=False)
            else:
                dist.all_gather(out, inp)
        torch.cuda.synchronize()

        return (time.perf_counter() - start) * 1000 / iters

    def profile_reduce_scatter(self, numel, iters=50):
        inp = [torch.randn(numel, device=self.device) for _ in range(self.world_size)]
        out = torch.empty(numel, device=self.device)

        for _ in range(10):
            if self.use_torchcomms:
                self.comm.reduce_scatter(out, inp, ReduceOp.SUM, async_op=False)
            else:
                dist.reduce_scatter(out, inp)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            if self.use_torchcomms:
                self.comm.reduce_scatter(out, inp, ReduceOp.SUM, async_op=False)
            else:
                dist.reduce_scatter(out, inp)
        torch.cuda.synchronize()

        return (time.perf_counter() - start) * 1000 / iters

    def run(self, tensor_sizes: List[int]):
        results = []

        header = f"\n{'='*80}\n"
        header += f"Configuration: World Size={self.world_size}\n"
        header += f"Backend: {self.backend} ({'torchcomms' if self.use_torchcomms else 'torch.distributed'})\n"
        header += f"{'='*80}\n"

        if self.rank == 0:
            print(header, end='')
            results.append(header)

        for n in tensor_sizes:
            size_mb = n * 4 / 1024 / 1024  # float32 = 4 bytes
            size_gb = size_mb / 1024

            ar_time = self.profile_all_reduce(n)
            ag_time = self.profile_all_gather(n)
            rs_time = self.profile_reduce_scatter(n)

            # Calculate bandwidth metrics
            # AllReduce: algbw = S * 2 * (N-1) / N, busbw = S / time
            ar_algbw = size_gb * 2 * (self.world_size - 1) / self.world_size / (ar_time / 1000)
            ar_busbw = size_gb / (ar_time / 1000)

            # AllGather: algbw = S * (N-1) / N, busbw = S * (N-1) / (N * time)
            ag_algbw = size_gb * (self.world_size - 1) / self.world_size / (ag_time / 1000)
            ag_busbw = size_gb * (self.world_size - 1) / self.world_size / (ag_time / 1000)

            # ReduceScatter: algbw = S * (N-1) / N, busbw = S * (N-1) / (N * time)
            rs_algbw = size_gb * (self.world_size - 1) / self.world_size / (rs_time / 1000)
            rs_busbw = size_gb * (self.world_size - 1) / self.world_size / (rs_time / 1000)

            result = f"\nTensor Size: {n:,} elements ({size_mb:.2f} MB)\n"
            result += f"  AllReduce:     {ar_time:.3f} ms | AlgBW: {ar_algbw:.2f} GB/s | BusBW: {ar_busbw:.2f} GB/s\n"
            result += f"  AllGather:     {ag_time:.3f} ms | AlgBW: {ag_algbw:.2f} GB/s | BusBW: {ag_busbw:.2f} GB/s\n"
            result += f"  ReduceScatter: {rs_time:.3f} ms | AlgBW: {rs_algbw:.2f} GB/s | BusBW: {rs_busbw:.2f} GB/s\n"

            if self.rank == 0:
                print(result, end='')
                results.append(result)

        footer = f"\n{'='*80}\nProfiling Complete!\n{'='*80}\n"
        if self.rank == 0:
            print(footer, end='')
            results.append(footer)
            print(f"\nResults saved to: {self.output_file}")
            with open(self.output_file, "w") as f:
                f.writelines(results)

    def finalize(self):
        if self.use_torchcomms:
            self.comm.finalize()
        else:
            dist.barrier()
            dist.destroy_process_group()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Profile collective communications")
    parser.add_argument("--backend", default="nccl", choices=["nccl", "ncclx", "gloo"], 
                       help="Communication backend")
    parser.add_argument(
        "--tensor-sizes",
        nargs="+",
        type=int,
        default=[1_048_576, 4_194_304, 16_777_216],
        help="Tensor sizes in number of elements (default: 1M, 4M, 16M)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save profiling results"
    )
    args = parser.parse_args()

    # Check if ncclx is requested but torchcomms not available
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    if args.backend == "ncclx" and not TORCHCOMMS_AVAILABLE:
        if rank == 0:
            print("ERROR: ncclx backend requested but torchcomms is not installed.")
            print("Please install torchcomms or use 'nccl' backend instead.")
        exit(1)

    # Create output directory if it doesn't exist
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir,
        f"profile_{args.backend}_world{world_size}_{timestamp}.txt"
    )

    profiler = CollectiveProfiler(args.backend, output_file)
    profiler.run(args.tensor_sizes)
    profiler.finalize()


if __name__ == "__main__":
    main()