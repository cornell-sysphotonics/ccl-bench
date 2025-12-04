#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正确的带宽计算脚本 - 基于 α-β 模型和 Ring AllReduce 理论

============================================================================
核心公式（来自课程）：
============================================================================

1. α-β 模型：
   T = α + s·β
   其中：
     - T: 通信时间 (seconds)
     - α: 启动延迟 (seconds)
     - s: 发送的数据量 (bytes)
     - β: 每字节传输时间 (seconds/byte)，β ≈ 1/bandwidth

2. Ring AllReduce 的通信量：
   每个 GPU 发送的总数据量: bytes_per_gpu = 2·(n-1)·s/n
   
   其中：
     - n: 参与通信的 GPU 数量
     - s: 待 AllReduce 的 tensor 总大小 (bytes)
   
   时间模型: T_allreduce = 2(n-1)α + 2(n-1)·s/(n·B)

3. 各种 collective 的每 GPU 发送数据量:
   - AllReduce:     bytes_per_gpu = 2·(n-1)·s/n  (scatter-reduce + allgather)
   - AllGather:     bytes_per_gpu = (n-1)·s/n    (每个 GPU 发送自己的 shard)
   - ReduceScatter: bytes_per_gpu = (n-1)·s/n    (每个 GPU 发送数据并做 reduce)
   - Broadcast:     bytes_per_gpu = s            (root 发送整个 tensor)
   - Send/Recv:     bytes_per_gpu = s            (点对点)

4. 带宽计算:
   bandwidth = bytes_per_gpu / T_measured

============================================================================
重要说明：
============================================================================
nsys 不直接提供 NCCL 调用的字节数！
因此我们需要：
1. 从模型配置推断 tensor size（需要用户提供）
2. 或者从已知的参数量和调用次数反推

本脚本支持两种模式：
- 模式 A: 用户提供模型配置（精确）
- 模式 B: 用户提供总参数量，脚本根据调用次数估算（近似）

============================================================================
"""

import subprocess
import os
import sqlite3
from collections import defaultdict
import numpy as np


# ============================================================================
# 硬件配置 (Perlmutter A100)
# ============================================================================
# NVLink (节点内, GPU-to-GPU):
#   - NV4 = 4 NVLinks × 25 GB/s = 100 GB/s 单向
# Slingshot (节点间):
#   - 4 HSN NICs × 25 GB/s = 100 GB/s 总带宽
#   - 单 NIC: 25 GB/s
HARDWARE_BW_NVLINK = 100e9      # 100 GB/s (NVLink NV4, 单向)
HARDWARE_BW_SLINGSHOT = 25e9    # 25 GB/s (单个 Slingshot NIC)


class ModelConfig:
    """模型配置 - 用于计算 tensor size"""
    
    def __init__(self, model_name, hidden_size, num_layers, num_attention_heads,
                 num_key_value_heads, intermediate_size, vocab_size,
                 batch_size=1, seq_length=4096, dtype_bytes=2):
        self.model_name = model_name
        self.hidden_size = hidden_size           # 隐藏层维度 (e.g., 4096 for Llama-8B)
        self.num_layers = num_layers             # 层数 (e.g., 32 for Llama-8B)
        self.num_attention_heads = num_attention_heads   # 注意力头数 (e.g., 32)
        self.num_key_value_heads = num_key_value_heads   # KV 头数 (e.g., 8 for GQA)
        self.intermediate_size = intermediate_size       # FFN 中间层维度 (e.g., 14336)
        self.vocab_size = vocab_size             # 词表大小 (e.g., 128256)
        self.batch_size = batch_size             # micro batch size
        self.seq_length = seq_length             # 序列长度
        self.dtype_bytes = dtype_bytes           # 数据类型字节数 (bf16=2, fp32=4)
    
    @property
    def total_params(self):
        """估算模型总参数量"""
        # Embedding
        embed_params = self.vocab_size * self.hidden_size
        
        # 每层参数
        # Attention: Q, K, V, O projections
        head_dim = self.hidden_size // self.num_attention_heads
        q_params = self.hidden_size * self.hidden_size
        k_params = self.hidden_size * self.num_key_value_heads * head_dim
        v_params = self.hidden_size * self.num_key_value_heads * head_dim
        o_params = self.hidden_size * self.hidden_size
        attn_params = q_params + k_params + v_params + o_params
        
        # FFN: gate, up, down projections (for SwiGLU)
        ffn_params = self.hidden_size * self.intermediate_size * 3
        
        # LayerNorm (negligible)
        norm_params = self.hidden_size * 2 * 2  # 2 norms per layer
        
        layer_params = attn_params + ffn_params + norm_params
        total_layer_params = layer_params * self.num_layers
        
        # Output embedding (usually tied with input)
        output_params = self.vocab_size * self.hidden_size
        
        return embed_params + total_layer_params + output_params


class ParallelConfig:
    """并行配置"""
    
    def __init__(self, dp_size=1, tp_size=1, pp_size=1, ep_size=1, zero_stage=3):
        self.dp_size = dp_size    # Data Parallel size
        self.tp_size = tp_size    # Tensor Parallel size
        self.pp_size = pp_size    # Pipeline Parallel size
        self.ep_size = ep_size    # Expert Parallel size (for MoE)
        self.zero_stage = zero_stage  # ZeRO stage (0, 1, 2, 3)
    
    @property
    def world_size(self):
        return self.dp_size * self.tp_size * self.pp_size


# ============================================================================
# 预定义模型配置
# ============================================================================
PREDEFINED_MODELS = {
    "llama-3.1-8b": ModelConfig(
        model_name="Llama-3.1-8B",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
        batch_size=1,
        seq_length=4096,
        dtype_bytes=2,
    ),
    "llama-3.1-70b": ModelConfig(
        model_name="Llama-3.1-70B",
        hidden_size=8192,
        num_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=28672,
        vocab_size=128256,
        batch_size=1,
        seq_length=4096,
        dtype_bytes=2,
    ),
    "deepseek-v2-lite": ModelConfig(
        model_name="DeepSeek-V2-Lite",
        hidden_size=2048,
        num_layers=27,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=10944,
        vocab_size=102400,
        batch_size=1,
        seq_length=4096,
        dtype_bytes=2,
    ),
    "qwen-32b": ModelConfig(
        model_name="Qwen-32B",
        hidden_size=5120,
        num_layers=64,
        num_attention_heads=40,
        num_key_value_heads=8,
        intermediate_size=27648,
        vocab_size=152064,
        batch_size=1,
        seq_length=4096,
        dtype_bytes=2,
    ),
}


def get_model_config(model_name):
    """获取模型配置"""
    return PREDEFINED_MODELS.get(model_name)


# ============================================================================
# Collective 通信量公式
# ============================================================================
def calc_bytes_per_gpu_ring_allreduce(tensor_size_bytes: int, n: int) -> int:
    """
    Ring AllReduce: 每个 GPU 发送的数据量
    
    公式: bytes_per_gpu = 2·(n-1)·s/n
    
    解释:
    - Phase 1 (scatter-reduce): 每个 GPU 发送 (n-1)·s/n bytes
    - Phase 2 (allgather): 每个 GPU 发送 (n-1)·s/n bytes
    - 总计: 2·(n-1)·s/n bytes
    """
    if n <= 1:
        return 0
    return int(2 * (n - 1) * tensor_size_bytes / n)


def calc_bytes_per_gpu_ring_allgather(tensor_size_bytes: int, n: int) -> int:
    """
    Ring AllGather: 每个 GPU 发送的数据量
    
    公式: bytes_per_gpu = (n-1)·s/n
    
    解释:
    - 每个 GPU 有自己的 shard (s/n bytes)
    - 需要发送给其他 n-1 个 GPU
    - 使用 Ring 算法: 每步发送一个 chunk，共 n-1 步
    - 总发送量: (n-1) × (s/n) = (n-1)·s/n bytes
    
    注意: 这里 s 是 **最终完整 tensor 的大小**
    """
    if n <= 1:
        return 0
    return int((n - 1) * tensor_size_bytes / n)


def calc_bytes_per_gpu_ring_reducescatter(tensor_size_bytes: int, n: int) -> int:
    """
    Ring ReduceScatter: 每个 GPU 发送的数据量
    
    公式: bytes_per_gpu = (n-1)·s/n
    
    解释:
    - 输入: 每个 GPU 有完整 tensor (s bytes)
    - 输出: 每个 GPU 得到一个 shard (s/n bytes)
    - 使用 Ring 算法: 每步发送一个 chunk 并做 reduce
    - 总发送量: (n-1) × (s/n) = (n-1)·s/n bytes
    
    注意: 这里 s 是 **输入 tensor 的大小**
    """
    if n <= 1:
        return 0
    return int((n - 1) * tensor_size_bytes / n)


def calc_bytes_per_gpu_broadcast(tensor_size_bytes: int, n: int) -> int:
    """
    Broadcast: 发送方发送的数据量
    
    公式: bytes = s (只有 root 发送)
    
    从带宽角度，我们关心的是 root 的发送量
    """
    return tensor_size_bytes


def calc_bytes_per_gpu_p2p(tensor_size_bytes: int) -> int:
    """
    Point-to-Point (Send/Recv): 发送方发送的数据量
    
    公式: bytes = s
    """
    return tensor_size_bytes


# ============================================================================
# 带宽计算
# ============================================================================
def calc_bandwidth(bytes_per_gpu: int, duration_s: float) -> float:
    """
    计算带宽
    
    公式: bandwidth = bytes_per_gpu / T
    
    Args:
        bytes_per_gpu: 每个 GPU 发送的数据量 (bytes)
        duration_s: 通信时间 (seconds)
    
    Returns:
        float: 带宽 (bytes/second)
    """
    if duration_s > 0:
        return bytes_per_gpu / duration_s
    return 0.0


def calc_utilization(bandwidth: float, hardware_bw: float) -> float:
    """计算带宽利用率"""
    if hardware_bw > 0:
        return bandwidth / hardware_bw
    return 0.0


# ============================================================================
# 从 trace 提取 NCCL 事件
# ============================================================================
def get_collective_type(kernel_name: str) -> str:
    """从 kernel 名称获取 collective 类型"""
    name_lower = kernel_name.lower()
    
    if 'allreduce' in name_lower:
        return 'AllReduce'
    elif 'reducescatter' in name_lower:
        return 'ReduceScatter'
    elif 'allgather' in name_lower:
        return 'AllGather'
    elif 'broadcast' in name_lower:
        return 'Broadcast'
    elif 'alltoall' in name_lower:
        return 'AllToAll'
    elif 'send' in name_lower:
        return 'Send'
    elif 'recv' in name_lower:
        return 'Recv'
    else:
        return 'Other'


def categorize_by_parallelism(kernel_name: str, parallel_config: ParallelConfig) -> str:
    """
    根据 kernel 类型和并行配置，判断属于哪种并行通信
    
    对于 ZeRO-3:
    - ReduceScatter: DP (梯度分片)
    - AllGather: DP (参数收集)
    - AllReduce: DP (loss scale 等) 或 TP (tensor parallel reduce)
    - Send/Recv: PP (pipeline)
    - AllToAll: EP (expert parallel)
    """
    name_lower = kernel_name.lower()
    
    if 'send' in name_lower or 'recv' in name_lower:
        return 'PP'
    if 'alltoall' in name_lower:
        return 'EP'
    if 'reducescatter' in name_lower:
        return 'DP'
    if 'allgather' in name_lower:
        return 'DP'
    if 'allreduce' in name_lower:
        # 如果有 TP，可能是 TP AllReduce
        # 简化处理: 默认为 DP
        return 'DP'
    if 'broadcast' in name_lower:
        return 'OTHER'
    return 'OTHER'


def extract_nccl_events(sqlite_file):
    """从 SQLite 文件提取 NCCL 事件"""
    
    if not os.path.exists(sqlite_file):
        print(f"Error: SQLite file not found: {sqlite_file}")
        return []
    
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()
    
    # 查询所有 NCCL kernels
    query = """
    SELECT k.start, k.end, k.streamId, k.deviceId, s.value as name
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName = s.id
    WHERE s.value LIKE '%nccl%'
    ORDER BY k.start
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    events = []
    for start, end, stream_id, device_id, name in rows:
        duration_ns = end - start
        collective_type = get_collective_type(name)
        
        events.append({
            'start_ns': start,
            'end_ns': end,
            'duration_ns': duration_ns,
            'duration_s': duration_ns * 1e-9,
            'stream_id': stream_id,
            'device_id': device_id,
            'kernel_name': name,
            'collective_type': collective_type,
        })
    
    return events


# ============================================================================
# 核心分析函数
# ============================================================================
def analyze_bandwidth_correct(
    nsys_file,
    model_config,
    parallel_config,
    hardware_bw=HARDWARE_BW_NVLINK,
    num_iterations=None,
    allgather_tensor_mb=None,
    reducescatter_tensor_mb=None,
):
    """
    使用正确的公式分析带宽
    
    核心思路:
    1. 从 trace 获取每个 NCCL 调用的时间
    2. 根据 collective 类型和模型配置，计算 tensor size
    3. 使用正确的 Ring 模型公式计算每个 GPU 发送的数据量
    4. 计算带宽: bandwidth = bytes_per_gpu / T
    
    Args:
        nsys_file: nsys-rep 或 sqlite 文件路径
        model_config: 模型配置
        parallel_config: 并行配置
        hardware_bw: 硬件带宽 (bytes/s)
        num_iterations: 迭代次数 (用于计算每次调用的 tensor size)
    """
    # 1. 获取 SQLite 文件路径
    if nsys_file.endswith('.nsys-rep'):
        sqlite_file = nsys_file.replace('.nsys-rep', '.sqlite')
    else:
        sqlite_file = nsys_file
    
    if not os.path.exists(sqlite_file):
        print(f"Error: SQLite file not found: {sqlite_file}")
        print(f"Please export the nsys-rep file first:")
        print(f"  nsys export --type=sqlite {nsys_file}")
        return None
    
    print(f"\n{'='*80}")
    print("正确的带宽分析 (基于 α-β 模型和 Ring AllReduce 理论)")
    print(f"{'='*80}")
    print(f"Trace 文件: {nsys_file}")
    print(f"模型: {model_config.model_name}")
    print(f"模型参数量: {model_config.total_params / 1e9:.2f}B")
    print(f"数据类型: {'bf16' if model_config.dtype_bytes == 2 else 'fp32'}")
    print(f"并行配置: DP={parallel_config.dp_size}, TP={parallel_config.tp_size}, "
          f"PP={parallel_config.pp_size}, EP={parallel_config.ep_size}")
    print(f"ZeRO stage: {parallel_config.zero_stage}")
    print(f"硬件带宽: {hardware_bw/1e9:.1f} GB/s")
    
    # 2. 提取 NCCL 事件
    events = extract_nccl_events(sqlite_file)
    if not events:
        print("No NCCL events found!")
        return None
    
    print(f"\n总 NCCL 事件数: {len(events)}")
    
    # 3. 统计各类 collective 的数量
    collective_counts = defaultdict(int)
    for e in events:
        collective_counts[e['collective_type']] += 1
    
    print(f"\nCollective 类型统计:")
    for coll_type, count in sorted(collective_counts.items()):
        print(f"  {coll_type}: {count}")
    
    # 4. 计算每种 collective 的 tensor size
    # 
    # 核心问题：nsys 不提供 NCCL 调用的字节数！
    # 我们需要从模型配置和调用模式来推断。
    #
    # 对于 ZeRO-3:
    # - AllGather: 在前向时收集参数，每个参数组单独调用
    # - ReduceScatter: 在后向时分发梯度
    #
    # 更准确的估算方法：
    # 1. 统计每个 iteration 的调用次数
    # 2. 根据模型结构估算每个参数组的大小
    #
    total_params = model_config.total_params
    total_bytes = total_params * model_config.dtype_bytes
    dp_size = parallel_config.dp_size
    
    # 如果没有提供 num_iterations，从 trace 推断
    if num_iterations is None:
        num_iterations = 50  # 默认值，需要用户确认
    
    allgather_count = collective_counts.get('AllGather', 0)
    reducescatter_count = collective_counts.get('ReduceScatter', 0)
    allreduce_count = collective_counts.get('AllReduce', 0)
    broadcast_count = collective_counts.get('Broadcast', 0)
    
    # 更准确的 tensor size 估算
    # 
    # 对于 ZeRO-3：
    # - 每个 iteration 前向时会对所有参数做 AllGather（分组进行）
    # - 每个 iteration 后向时会对所有梯度做 ReduceScatter
    #
    # 但注意：我们测量的是 **单个 GPU 发送的数据量**，不是 tensor 大小！
    # 
    # 正确的方法是：
    # - 总传输字节 = sum(每次调用的 bytes_per_gpu)
    # - 对于 AllGather: bytes_per_gpu = (n-1)/n × tensor_size
    # - 所以: 总传输字节 = (n-1)/n × 总 tensor_size 
    #       = (n-1)/n × total_bytes × iterations (前向+后向各一次)
    #
    # 但这仍然是循环论证...
    #
    # 替代方案：直接从 kernel 持续时间来反推，假设一定的带宽利用率
    # 或者：用户提供每次调用的实际 tensor size
    #
    # 这里我们用一个折中方案：
    # - 计算 "如果达到 X% 利用率，实际传输了多少数据"
    # - 然后报告不同假设下的带宽
    
    # 确定 tensor size
    # 
    # 优先使用用户指定的值，否则基于模型参数量估算
    #
    if allgather_tensor_mb is not None:
        # 用户指定
        avg_allgather_tensor_bytes = allgather_tensor_mb * 1e6
        print(f"\n使用用户指定的 AllGather tensor size: {allgather_tensor_mb:.2f} MB")
    elif allgather_count > 0:
        # 基于模型参数量估算
        calls_per_iter = allgather_count / num_iterations
        avg_allgather_tensor_bytes = total_bytes / calls_per_iter
        print(f"\n估算的 AllGather tensor size: {avg_allgather_tensor_bytes/1e6:.2f} MB (基于模型参数量)")
    else:
        avg_allgather_tensor_bytes = 0
    
    if reducescatter_tensor_mb is not None:
        # 用户指定
        avg_reducescatter_tensor_bytes = reducescatter_tensor_mb * 1e6
        print(f"使用用户指定的 ReduceScatter tensor size: {reducescatter_tensor_mb:.2f} MB")
    elif reducescatter_count > 0:
        # 基于模型参数量估算
        calls_per_iter = reducescatter_count / num_iterations
        avg_reducescatter_tensor_bytes = total_bytes / calls_per_iter
        print(f"估算的 ReduceScatter tensor size: {avg_reducescatter_tensor_bytes/1e6:.2f} MB (基于模型参数量)")
    else:
        avg_reducescatter_tensor_bytes = 0
    
    if allgather_tensor_mb is None and reducescatter_tensor_mb is None:
        print(f"\n注意: tensor size 是基于模型参数量估算的，可能不准确!")
        print(f"      如果利用率超过100%，请使用 --allgather-tensor-mb 和 --reducescatter-tensor-mb 指定实际值")
    
    # 5. 计算每个事件的带宽
    # 
    # 步骤（按照课程方法）：
    # 1. 从 trace 获取通信时间 T
    # 2. 根据 collective 类型和 tensor size s，计算每个 GPU 发送的数据量
    # 3. 带宽 = bytes_per_gpu / T
    #
    results_by_collective = defaultdict(list)
    results_by_parallelism = defaultdict(list)
    
    for e in events:
        coll_type = e['collective_type']
        parallelism = categorize_by_parallelism(e['kernel_name'], parallel_config)
        duration_s = e['duration_s']
        
        # 步骤 1: 确定 tensor size (s)
        if coll_type == 'AllReduce':
            # AllReduce 通常用于小规模同步（loss scale 等）
            tensor_size = model_config.hidden_size * model_config.dtype_bytes
        elif coll_type == 'AllGather':
            tensor_size = avg_allgather_tensor_bytes
        elif coll_type == 'ReduceScatter':
            tensor_size = avg_reducescatter_tensor_bytes
        elif coll_type == 'Broadcast':
            tensor_size = total_bytes / broadcast_count if broadcast_count > 0 else 0
        elif coll_type in ['Send', 'Recv']:
            # PP 通信：activation tensor
            tensor_size = (model_config.batch_size * model_config.seq_length * 
                          model_config.hidden_size * model_config.dtype_bytes)
        else:
            tensor_size = 0
        
        # 步骤 2: 计算每个 GPU 发送的数据量 (bytes_per_gpu)
        if coll_type == 'AllReduce':
            bytes_per_gpu = calc_bytes_per_gpu_ring_allreduce(tensor_size, dp_size)
        elif coll_type == 'AllGather':
            bytes_per_gpu = calc_bytes_per_gpu_ring_allgather(tensor_size, dp_size)
        elif coll_type == 'ReduceScatter':
            bytes_per_gpu = calc_bytes_per_gpu_ring_reducescatter(tensor_size, dp_size)
        elif coll_type == 'Broadcast':
            bytes_per_gpu = calc_bytes_per_gpu_broadcast(tensor_size, dp_size)
        elif coll_type in ['Send', 'Recv']:
            bytes_per_gpu = calc_bytes_per_gpu_p2p(tensor_size)
        else:
            bytes_per_gpu = 0
        
        # 步骤 3: 计算带宽 = bytes_per_gpu / T
        bandwidth = calc_bandwidth(bytes_per_gpu, duration_s)
        utilization = calc_utilization(bandwidth, hardware_bw)
        
        result = {
            'duration_s': duration_s,
            'duration_us': duration_s * 1e6,
            'tensor_size_bytes': tensor_size,
            'bytes_per_gpu': bytes_per_gpu,
            'bandwidth_gbps': bandwidth / 1e9,
            'utilization_pct': utilization * 100,
        }
        
        results_by_collective[coll_type].append(result)
        results_by_parallelism[parallelism].append(result)
    
    # 6. 汇总统计
    stats = {
        'model': model_config.model_name,
        'total_params': model_config.total_params,
        'parallel_config': {
            'dp_size': parallel_config.dp_size,
            'tp_size': parallel_config.tp_size,
            'pp_size': parallel_config.pp_size,
            'ep_size': parallel_config.ep_size,
            'zero_stage': parallel_config.zero_stage,
        },
        'hardware_bw_gbps': hardware_bw / 1e9,
        'num_iterations': num_iterations,
        'total_events': len(events),
        'per_collective': {},
        'per_parallelism': {},
    }
    
    # 按 collective 类型汇总
    print(f"\n{'='*80}")
    print("带宽分析结果 (按 Collective 类型)")
    print(f"{'='*80}")
    print(f"{'Collective':<15} {'Events':>8} {'Bytes(GB)':>12} {'avg_bw':>10} {'p50_bw':>10} {'avg_util':>10}")
    print("-" * 70)
    
    for coll_type in ['AllReduce', 'AllGather', 'ReduceScatter', 'Broadcast', 'AllToAll', 'Send', 'Recv', 'Other']:
        results = results_by_collective.get(coll_type, [])
        if results:
            bw_list = [r['bandwidth_gbps'] for r in results]
            util_list = [r['utilization_pct'] for r in results]
            bytes_list = [r['bytes_per_gpu'] for r in results]
            
            stats['per_collective'][coll_type] = {
                'num_events': len(results),
                'total_bytes_gb': sum(bytes_list) / 1e9,
                'avg_bw_gbps': np.mean(bw_list),
                'p50_bw_gbps': np.percentile(bw_list, 50),
                'p95_bw_gbps': np.percentile(bw_list, 95),
                'avg_util_pct': np.mean(util_list),
                'p50_util_pct': np.percentile(util_list, 50),
                'p95_util_pct': np.percentile(util_list, 95),
            }
            
            print(f"{coll_type:<15} {len(results):>8} {sum(bytes_list)/1e9:>12.2f} "
                  f"{np.mean(bw_list):>10.2f} {np.percentile(bw_list, 50):>10.2f} "
                  f"{np.mean(util_list):>9.1f}%")
    
    # 按并行类型汇总
    print(f"\n{'='*80}")
    print("带宽分析结果 (按并行类型)")
    print(f"{'='*80}")
    print(f"{'Parallelism':<15} {'Events':>8} {'Bytes(GB)':>12} {'avg_bw':>10} {'p50_bw':>10} {'avg_util':>10}")
    print("-" * 70)
    
    for para_type in ['DP', 'TP', 'PP', 'EP', 'OTHER']:
        results = results_by_parallelism.get(para_type, [])
        if results:
            bw_list = [r['bandwidth_gbps'] for r in results]
            util_list = [r['utilization_pct'] for r in results]
            bytes_list = [r['bytes_per_gpu'] for r in results]
            
            stats['per_parallelism'][para_type] = {
                'num_events': len(results),
                'total_bytes_gb': sum(bytes_list) / 1e9,
                'avg_bw_gbps': np.mean(bw_list),
                'p50_bw_gbps': np.percentile(bw_list, 50),
                'p95_bw_gbps': np.percentile(bw_list, 95),
                'avg_util_pct': np.mean(util_list),
                'p50_util_pct': np.percentile(util_list, 50),
                'p95_util_pct': np.percentile(util_list, 95),
            }
            
            print(f"{para_type:<15} {len(results):>8} {sum(bytes_list)/1e9:>12.2f} "
                  f"{np.mean(bw_list):>10.2f} {np.percentile(bw_list, 50):>10.2f} "
                  f"{np.mean(util_list):>9.1f}%")
    
    # 全局汇总
    all_results = []
    for results in results_by_collective.values():
        all_results.extend(results)
    
    if all_results:
        all_bw = [r['bandwidth_gbps'] for r in all_results]
        all_util = [r['utilization_pct'] for r in all_results]
        all_bytes = [r['bytes_per_gpu'] for r in all_results]
        
        stats['global'] = {
            'total_events': len(all_results),
            'total_bytes_gb': sum(all_bytes) / 1e9,
            'avg_bw_gbps': np.mean(all_bw),
            'p50_bw_gbps': np.percentile(all_bw, 50),
            'p95_bw_gbps': np.percentile(all_bw, 95),
            'avg_util_pct': np.mean(all_util),
            'p50_util_pct': np.percentile(all_util, 50),
            'p95_util_pct': np.percentile(all_util, 95),
        }
        
        print("-" * 70)
        print(f"{'TOTAL':<15} {len(all_results):>8} {sum(all_bytes)/1e9:>12.2f} "
              f"{np.mean(all_bw):>10.2f} {np.percentile(all_bw, 50):>10.2f} "
              f"{np.mean(all_util):>9.1f}%")
    
    print(f"{'='*80}")
    
    # 打印公式说明
    print(f"\n使用的带宽计算公式 (Ring AllReduce 模型):")
    print(f"  AllReduce:     bytes_per_gpu = 2·(n-1)·s/n")
    print(f"  AllGather:     bytes_per_gpu = (n-1)·s/n")
    print(f"  ReduceScatter: bytes_per_gpu = (n-1)·s/n")
    print(f"  Broadcast:     bytes_per_gpu = s")
    print(f"  Send/Recv:     bytes_per_gpu = s")
    print(f"")
    print(f"  bandwidth = bytes_per_gpu / T")
    print(f"  utilization = bandwidth / hardware_bw")
    print(f"")
    print(f"  其中: n = {dp_size} (DP size), hardware_bw = {hardware_bw/1e9:.0f} GB/s")
    
    return stats


# ============================================================================
# CLI 入口
# ============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='正确的带宽分析 - 基于 α-β 模型和 Ring AllReduce 理论',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析 Llama-3.1-8B，DP=4 (自动估算 tensor size)
  python bandwidth_correct_analyzer.py nsys_1node/llama_3.1_8b_trace_50.sqlite \\
      --model llama-3.1-8b --dp 4 --iterations 50
  
  # 分析 Llama-3.1-8B，DP=4 (手动指定 tensor size)
  python bandwidth_correct_analyzer.py nsys_1node/llama_3.1_8b_trace_50.sqlite \\
      --model llama-3.1-8b --dp 4 --iterations 50 \\
      --allgather-tensor-mb 8.0 --reducescatter-tensor-mb 8.0
  
  # 分析 Llama-3.1-8B，DP=8 (2节点，节点间带宽 25 GB/s)
  python bandwidth_correct_analyzer.py nsys_2node/trace_2nodes_rank_0_50.sqlite \\
      --model llama-3.1-8b --dp 8 --iterations 50 --hardware-bw 25
      
公式说明 (Ring AllReduce 模型):
  AllReduce:     bytes_per_gpu = 2·(n-1)·s/n
  AllGather:     bytes_per_gpu = (n-1)·s/n
  ReduceScatter: bytes_per_gpu = (n-1)·s/n
  Broadcast:     bytes_per_gpu = s
  Send/Recv:     bytes_per_gpu = s
  
  带宽: bandwidth = bytes_per_gpu / T
  利用率: utilization = bandwidth / hardware_bw
        """)
    
    parser.add_argument('trace_file', help='nsys-rep 或 sqlite 文件路径')
    parser.add_argument('--model', choices=list(PREDEFINED_MODELS.keys()),
                       default='llama-3.1-8b', help='模型名称')
    parser.add_argument('--dp', type=int, default=4, help='Data Parallel size')
    parser.add_argument('--tp', type=int, default=1, help='Tensor Parallel size')
    parser.add_argument('--pp', type=int, default=1, help='Pipeline Parallel size')
    parser.add_argument('--ep', type=int, default=1, help='Expert Parallel size')
    parser.add_argument('--zero', type=int, default=3, help='ZeRO stage (0, 1, 2, 3)')
    parser.add_argument('--iterations', type=int, default=50, help='训练迭代次数')
    parser.add_argument('--hardware-bw', type=float, default=100.0,
                       help='硬件带宽 (GB/s)，节点内=100, 节点间=25')
    parser.add_argument('--allgather-tensor-mb', type=float, default=None,
                       help='手动指定 AllGather 的平均 tensor size (MB)')
    parser.add_argument('--reducescatter-tensor-mb', type=float, default=None,
                       help='手动指定 ReduceScatter 的平均 tensor size (MB)')
    
    args = parser.parse_args()
    
    # 获取模型配置
    model_config = PREDEFINED_MODELS[args.model]
    
    # 创建并行配置
    parallel_config = ParallelConfig(
        dp_size=args.dp,
        tp_size=args.tp,
        pp_size=args.pp,
        ep_size=args.ep,
        zero_stage=args.zero,
    )
    
    # 运行分析
    stats = analyze_bandwidth_correct(
        args.trace_file,
        model_config,
        parallel_config,
        hardware_bw=args.hardware_bw * 1e9,
        num_iterations=args.iterations,
        allgather_tensor_mb=args.allgather_tensor_mb,
        reducescatter_tensor_mb=args.reducescatter_tensor_mb,
    )
    
    if stats:
        print(f"\n=== 分析完成 ===")
        if stats.get('global'):
            g = stats['global']
            print(f"总事件数: {g['total_events']}")
            print(f"总传输数据: {g['total_bytes_gb']:.2f} GB")
            print(f"平均带宽: {g['avg_bw_gbps']:.2f} GB/s")
            print(f"平均利用率: {g['avg_util_pct']:.1f}%")


if __name__ == '__main__':
    main()

