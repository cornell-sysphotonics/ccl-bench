import os
import torch
import torch.distributed as dist
import time
import argparse

def benchmark_op(op_name, tensor, count, group):
    # Warmup
    for _ in range(5):
        if op_name == "all_reduce":
            dist.all_reduce(tensor, group=group)
        elif op_name == "all_gather":
            # 模拟 FSDP 的 all_gather_into_tensor
            output_tensor = torch.empty(tensor.numel() * dist.get_world_size(), dtype=tensor.dtype, device=tensor.device)
            dist.all_gather_into_tensor(output_tensor, tensor, group=group)

    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(count):
        if op_name == "all_reduce":
            dist.all_reduce(tensor, group=group)
        elif op_name == "all_gather":
            output_tensor = torch.empty(tensor.numel() * dist.get_world_size(), dtype=tensor.dtype, device=tensor.device)
            dist.all_gather_into_tensor(output_tensor, tensor, group=group)
            
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size_mb", type=int, default=128, help="Tensor size in MB")
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations")
    args = parser.parse_args()

    # 初始化 (即使是 MSCCL++ 插件，也要用 nccl 后端名)
    dist.init_process_group(backend="nccl")
    
    # === 关键修复 1: 正确获取 Local Rank ===
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # 尝试从环境变量获取 LOCAL_RANK，如果没有则通过计算获取
    local_rank = int(os.environ.get("LOCAL_RANK", rank % 4))
    
    # 设置当前设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # === 关键修复 2: 补回 num_elements 定义 ===
    # 计算 Float16 元素数量: Size(MB) * 1024^2 / 2 bytes
    num_elements = args.size_mb * 1024 * 1024 // 2
    
    # 创建 Tensor (使用正确的 device)
    tensor = torch.ones(num_elements, dtype=torch.float16, device=device)
    
    if rank == 0:
        print(f"=== Benchmarking {dist.get_backend()} on {world_size} GPUs ===")
        print(f"Tensor Size: {args.size_mb} MB ({num_elements} elements, FP16)")

    # Test 1: AllReduce (这是你 Proposal 的重点!)
    try:
        avg_time = benchmark_op("all_reduce", tensor, args.iters, dist.group.WORLD)
        # Algorithmic Bandwidth = Size / Time
        alg_bw = (args.size_mb / 1024) / avg_time # GB/s
        if rank == 0:
            print(f"[AllReduce] Avg Time: {avg_time*1000:.3f} ms | Bandwidth: {alg_bw:.2f} GB/s")
    except Exception as e:
        if rank == 0: print(f"[AllReduce] Failed: {e}")

    # Test 2: AllGather (测试你生成的 XML 是否生效)
    try:
        avg_time = benchmark_op("all_gather", tensor, args.iters, dist.group.WORLD)
        alg_bw = (args.size_mb / 1024) / avg_time 
        if rank == 0:
            print(f"[AllGather] Avg Time: {avg_time*1000:.3f} ms | Bandwidth: {alg_bw:.2f} GB/s")
    except Exception as e:
        if rank == 0: 
            print(f"[AllGather] Failed: {e}")
            print("Hint: If AllReduce works but AllGather fails, generated XML might mismtch FSDP requirements.")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()