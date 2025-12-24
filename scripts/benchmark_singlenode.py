"""
SIMPLE BENCHMARK with Nsight Systems profiling.
Tests NCCL vs NVSHMEM using torchrun with nsys profiling.
"""

import subprocess
import time
import json
import os
from datetime import datetime

# Config
TOML_FILE = "./torchtitan/models/llama3/train_configs/llama3_8bnew.toml"
NUM_GPUS = 4
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Environment
env = os.environ.copy()
env["PYTHONPATH"] = "/pscratch/sd/i/ishita/LLM-Benchmarking-NVSHMEM-NCCLGIN:" + env.get("PYTHONPATH", "")
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# Enable profiling (set to False for faster runs without profiling)
ENABLE_PROFILING = False

# Command templates
if ENABLE_PROFILING:
    # With nsys profiling - each rank gets its own profile

    CMD_TEMPLATE = (
        "nsys profile "
        "--force-overwrite=true "
        "--sample=none "
        "--trace=cuda,nvtx,mpi,osrt "
        "--cuda-memory-usage=true "
        "--stats=true "
        "-o {nsys_out} "
        "torchrun --nproc_per_node=4 "
        "torchtitan/train.py "
        "--job.config_file {toml}"
    )


else:
    # Without profiling - faster
    CMD_TEMPLATE = (
        "torchrun "
        "--nproc_per_node={nproc} "
        "--rdzv_backend=c10d "
        "--rdzv_endpoint=localhost:29500 "
        "torchtitan/train.py "
        "--job.config_file {toml}"
    )

def run_benchmark(backend):
    """Run training for one backend."""
    print(f"\n{'='*60}")
    print(f"Testing {backend.upper()}")
    print(f"{'='*60}\n")
    
    # Set backend
    env["COMM_BACKEND"] = backend
    
    # Output paths
    nsys_out = f"nsys_{backend}_{timestamp}"
    log_file = f"logs/bench_{backend}_{timestamp}.log"
    
    os.makedirs("logs", exist_ok=True)
    
    # Build command
    cmd = CMD_TEMPLATE.format(
        nproc=NUM_GPUS,
        nsys_out=nsys_out,
        toml=TOML_FILE,
    )
    
    print(f"Command: {cmd}")
    print(f"Log: {log_file}\n")
    
    # Run
    start = time.time()
    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    elapsed = time.time() - start
    
    # Parse results
    tokens_per_sec = []
    steps = 0
    success = result.returncode == 0
    
    try:
        with open(log_file, "r") as f:
            for line in f:
                if "tokens/sec:" in line:
                    try:
                        val = float(line.split("tokens/sec:")[-1].split()[0])
                        tokens_per_sec.append(val)
                    except:
                        pass
                if "step" in line.lower() and "loss" in line.lower():
                    steps += 1
    except FileNotFoundError:
        pass
    
    avg_throughput = sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0
    
    # Print results
    print(f"Results:")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Success: {success}")
    print(f"  Steps: {steps}")
    print(f"  Avg throughput: {avg_throughput:.1f} tokens/sec")
    
    if ENABLE_PROFILING:
        print(f"  Profile files:")
        for rank in range(NUM_GPUS):
            profile_file = f"{nsys_out}_rank{rank}.nsys-rep"
            if os.path.exists(profile_file):
                print(f"    - {profile_file}")
    
    return {
        "backend": backend,
        "success": success,
        "time_sec": round(elapsed, 2),
        "steps": steps,
        "avg_tokens_per_sec": round(avg_throughput, 1),
        "log_file": log_file,
        "profile_files": [
            f"{nsys_out}_rank{i}.nsys-rep" for i in range(NUM_GPUS)
        ] if ENABLE_PROFILING else [],
    }


def main():
    print("\n" + "="*60)
    print("NCCL vs NVSHMEM BENCHMARK")
    print("="*60)
    print(f"GPUs: {NUM_GPUS}")
    print(f"Config: {TOML_FILE}")
    print(f"Profiling: {ENABLE_PROFILING}")
    print("="*60)
    
    results = []
    
    # Test NCCL first (baseline)
    print("\n>>> Running NCCL baseline...")
    results.append(run_benchmark("nccl"))
    
    # Test NVSHMEM
    print("\n>>> Running NVSHMEM...")
    results.append(run_benchmark("nvshmem"))
    
    # Save results
    output = f"results_{timestamp}.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Final comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    nccl = next(r for r in results if r["backend"] == "nccl")
    nvshmem = next(r for r in results if r["backend"] == "nvshmem")
    
    print(f"NCCL:    {nccl['avg_tokens_per_sec']:.1f} tokens/sec ({nccl['time_sec']}s)")
    print(f"NVSHMEM: {nvshmem['avg_tokens_per_sec']:.1f} tokens/sec ({nvshmem['time_sec']}s)")
    
    if nccl['success'] and nvshmem['success'] and nccl['avg_tokens_per_sec'] > 0:
        speedup = nvshmem['avg_tokens_per_sec'] / nccl['avg_tokens_per_sec']
        print(f"\nSpeedup: {speedup:.2f}x")
    elif not nccl['success']:
        print("\n⚠ NCCL failed - check logs")
    elif not nvshmem['success']:
        print("\n⚠ NVSHMEM failed - check logs (may have fallen back to NCCL)")
    
    print(f"\nResults saved to: {output}")
    
    if ENABLE_PROFILING:
        print("\nProfile files generated:")
        print("  Open in Nsight Systems GUI to view NVTX markers")
        print("  Look for: data_loading, forward_pass, backward_pass, optimizer_step")


if __name__ == "__main__":
    main()