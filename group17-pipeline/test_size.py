import torch
import torch.distributed as dist
import time
import pandas as pd
import matplotlib.pyplot as plt
import cupy as cp  # <--- 新增：需要 cupy 处理 AllReduce6 的数据搬运
import os

from mscclpp_manager import MscclppManager, MscclppStreamCompat
# 需要导入 AllReduce6 以便在脚本中做类型判断
from mscclpp_op import MscclppAllReduce6

# 配置
sizes_mb = [1, 4, 16, 64, 256, 1024, 4096] # 测试这些大小 (MB)
warmup = 5
trials = 20

def run_benchmark(rank, world_size):
    # 初始化 NCCL
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    
    # 确保 CuPy 使用正确的设备
    cp.cuda.Device(rank).use()

    # 初始化 MSCCL++
    mscclpp_mgr = MscclppManager(rank, world_size)
    
    results = []

    for size_mb in sizes_mb:
        nelem = size_mb * 1024 * 1024 // 2 # float16 (2 bytes)
        tensor = torch.randn(nelem, dtype=torch.float16, device=device)
        size_bytes = nelem * 2
        
        # ==========================================
        # 1. 测试 NCCL
        # ==========================================
        for _ in range(warmup):
            dist.all_reduce(tensor)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(trials):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        nccl_avg_time = (time.time() - start) / trials
        
        # ==========================================
        # 2. 测试 MSCCL++ (使用 CUDA Graph)
        # ==========================================
        
        # A. 预热 Manager (触发 Lazy Initialization 和 算法选择)
        # 这一步非常重要，它会创建 op 并存入 operators 字典
        for _ in range(warmup):
            mscclpp_mgr.run_all_reduce(tensor)
        
        torch.cuda.synchronize()

        # B. 提取算子 (Bypass Python Overhead)
        nelem = tensor.numel()
        dtype = tensor.dtype
        key = f"{nelem}_{dtype}"
        
        if key not in mscclpp_mgr.operators:
            print(f"[Rank {rank}] Error: Operator for {key} not found! Warmup failed?")
            return

        op = mscclpp_mgr.operators[key]
        
        # 准备流对象
        current_stream = torch.cuda.current_stream()
        mscclpp_stream_obj = MscclppStreamCompat(current_stream.cuda_stream)
        
        # C. 定义单步执行逻辑 (闭包)
        # 这里的逻辑会被录制进 CUDA Graph
        def step_func():
            # 确保 CuPy 操作也跑在当前的 PyTorch 流上，否则 Graph 录制会失败
            with cp.cuda.ExternalStream(current_stream.cuda_stream):
                if isinstance(op, MscclppAllReduce6):
                    # NVLS 特有逻辑：显存拷贝
                    # 注意：为了性能，这些变量最好在外部准备好，但在 Graph 模式下，
                    # 只要 tensor 地址不变，这里录制的指针就是有效的。
                    op_mem = op.get_memory()
                    cp_tensor = cp.asarray(tensor) # Zero-copy view
                    op_mem[:] = cp_tensor          # Record Copy In
                    op(mscclpp_stream_obj)         # Record Kernel
                    cp_tensor[:] = op_mem          # Record Copy Out
                else:
                    # 普通 AllReduce1/2
                    op(mscclpp_stream_obj)

        # D. CUDA Graph 录制
        # 先跑一次 step_func 确保 CuPy 分配等初始化完成
        step_func()
        torch.cuda.synchronize()
        
        g = torch.cuda.CUDAGraph()
        
        # 开始录制：在此期间的所有 GPU 操作都不会立即执行，而是被存入 g
        with torch.cuda.graph(g):
            step_func()
        
        torch.cuda.synchronize()
        
        # E. 极速重放 (Hot Loop)
        start = time.time()
        for _ in range(trials):
            g.replay() # CPU 发射这个指令几乎 0 开销，GPU 自动执行录制好的一连串操作
            
        torch.cuda.synchronize()
        mscclpp_avg_time = (time.time() - start) / trials

        # ==========================================
        # 3. 计算结果
        # ==========================================
        
        # Formula: S * 2 * (n-1)/n / t
        factor = 2 * (world_size - 1) / world_size
        nccl_bw = (size_bytes * factor) / nccl_avg_time / 1e9 # GB/s
        mscclpp_bw = (size_bytes * factor) / mscclpp_avg_time / 1e9 # GB/s

        results.append({
            "Size (MB)": size_mb,
            "NCCL BW (GB/s)": nccl_bw,
            "MSCCL++ BW (GB/s)": mscclpp_bw
        })
        
        if rank == 0:
            algo_name = "AllReduce6(NVLS)" if isinstance(op, MscclppAllReduce6) else "AllReduce1/2"
            print(f"Size {size_mb}MB [{algo_name}]: NCCL={nccl_bw:.2f} GB/s, MSCCL++={mscclpp_bw:.2f} GB/s")

    # 画图 (只在 Rank 0)
    if rank == 0:
        df = pd.DataFrame(results)
        df.set_index("Size (MB)", inplace=True)
        
        # 绘图
        ax = df.plot(kind='line', marker='o', figsize=(10, 6))
        plt.title("AllReduce Bandwidth Scaling: NCCL vs MSCCL++ (CUDA Graph)")
        plt.ylabel("Effective Bus Bandwidth (GB/s)")
        plt.xlabel("Message Size (MB)")
        plt.grid(True)
        plt.savefig("bandwidth_scaling.png")
        print("图表已保存为 bandwidth_scaling.png")

    dist.destroy_process_group()

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    run_benchmark(rank, world_size)