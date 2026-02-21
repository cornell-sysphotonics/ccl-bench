"""
Fixed Comprehensive Analysis for DeepSeek V2-Lite
Correctly identifies MoE/Expert kernels and NCCL operations
"""

import yaml
import sqlite3
import pandas as pd
import re
from typing import Dict
from pathlib import Path

def parse_filename_config(filename):
    """Extract configuration from filename"""
    config = {}
    ep_match = re.search(r'ep(\d+)', filename)
    config['EP'] = int(ep_match.group(1)) if ep_match else None
    tp_match = re.search(r'tp(\d+)', filename)
    config['TP'] = int(tp_match.group(1)) if tp_match else None
    pp_match = re.search(r'pp(\d+)', filename)
    config['PP'] = int(pp_match.group(1)) if pp_match else None
    dp_match = re.search(r'dp(\d+)', filename)
    config['DP'] = int(dp_match.group(1)) if dp_match else None
    config['model'] = 'deepseek' if 'deepseek' in filename.lower() else \
                      'llama' if 'llama' in filename.lower() else \
                      'qwen' if 'qwen' in filename.lower() else 'unknown'
    return config

def classify_operation(op_name):
    """Classify operation type - FIXED for DeepSeek kernels"""
    op_lower = op_name.lower()

    # NCCL Communication operations
    if 'nccldevkernel' in op_lower or 'nccl' in op_lower:
        if 'gather' in op_lower:
            return 'AllGather', 'Communication'
        elif 'reduce' in op_lower:
            return 'AllReduce', 'Communication'
        elif 'scatter' in op_lower:
            return 'ReduceScatter', 'Communication'
        elif 'all' in op_lower:
            return 'AllToAll', 'Communication'
        else:
            return 'NCCL_Other', 'Communication'

    # Cross-device reduce (also communication)
    elif 'cross_device_reduce' in op_lower:
        return 'CrossDeviceReduce', 'Communication'

    # MoE/Expert kernels - FIXED patterns for DeepSeek
    elif any(pattern in op_lower for pattern in [
        'fused_moe_kernel',
        'moe_align_block_size',
        'moe_sum_reduce',
        'gathertopk',  # Expert selection
        'topk',
        'expert'
    ]):
        return 'Expert_MoE', 'Compute'

    # Attention operations
    elif any(pattern in op_lower for pattern in [
        'attention',
        'flash',
        '_fwd_kernel',  # FlashAttention kernels
        '_fwd_grouped_kernel',
        'batchqkapplyrotary',
        'flashinfer'
    ]):
        return 'Attention', 'Compute'

    # GEMM operations
    elif 'gemm' in op_lower or 'gemv' in op_lower:
        return 'GEMM', 'Compute'

    # Normalization
    elif 'norm' in op_lower or 'rmsnorm' in op_lower or 'layernorm' in op_lower:
        return 'Normalization', 'Compute'

    # Activation functions
    elif 'act_and_mul' in op_lower or 'activation' in op_lower:
        return 'Activation', 'Compute'

    # Elementwise operations
    elif 'elementwise' in op_lower or 'triton_poi' in op_lower or 'triton_red' in op_lower:
        return 'Elementwise', 'Compute'
    #how do i go through cuda api to get theroij cuda stream is capturing to cross reduce 2 stage and then look in the cuda to look at the memcopy and then see how long the all reduce
    # Memory operations
    elif any(pattern in op_lower for pattern in ['copy', 'cat', 'index', 'scatter', 'gather']):
        # But not MoE gatherTopK
        if 'gathertopk' not in op_lower:
            return 'Memory', 'Compute'

    # Reduction operations
    elif 'reduce' in op_lower and 'cross_device' not in op_lower:
        return 'Reduce', 'Compute'

    else:
        return 'Other', 'Compute'

def analyze_trace_comprehensive(filename, output_prefix=None):
    """Comprehensive analysis of nsys trace"""

    if output_prefix is None:
        output_prefix = filename.replace('.sqlite', '')

    config = parse_filename_config(filename)

    conn = sqlite3.connect(filename)

    # ========================================================================
    # LOAD ALL KERNELS
    # ========================================================================

    kernel_query = """
    SELECT
        k.start,
        k.end,
        (k.end - k.start) as duration_ns,
        s.value as kernel_name,
        k.deviceId,
        k.correlationId,
        k.gridX * k.gridY * k.gridZ as grid_size,
        k.blockX * k.blockY * k.blockZ as block_size,
        k.staticSharedMemory + k.dynamicSharedMemory as total_shared_mem,
        k.registersPerThread,
        k.localMemoryTotal
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName = s.id
    ORDER BY k.start
    """

    all_kernels = pd.read_sql_query(kernel_query, conn)

    # Classify all kernels
    classifications = [classify_operation(name) for name in all_kernels['kernel_name']]
    all_kernels['operation_type'] = [c[0] for c in classifications]
    all_kernels['category'] = [c[1] for c in classifications]

    all_kernels['duration_us'] = all_kernels['duration_ns'] / 1000
    all_kernels['duration_ms'] = all_kernels['duration_ns'] / 1e6
    all_kernels['time_s'] = (all_kernels['start'] - all_kernels['start'].min()) / 1e9

    # ========================================================================
    # LOAD MEMCPY
    # ========================================================================

    memcpy_query = """
    SELECT
        start,
        end,
        bytes,
        copyKind,
        (end - start) as duration_ns
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    WHERE bytes > 10000
    ORDER BY start
    """

    memcpy_data = pd.read_sql_query(memcpy_query, conn)
    memcpy_data['bandwidth_GB_s'] = (memcpy_data['bytes'] / (memcpy_data['duration_ns'] / 1e9)) / 1e9

    conn.close()

    # ========================================================================
    # BASIC METRICS
    # ========================================================================

    trace_start = all_kernels['start'].min()
    trace_end = all_kernels['end'].max()
    total_duration_s = (trace_end - trace_start) / 1e9

    num_gpus = all_kernels['deviceId'].nunique()

    results = {
        'filename': filename,
        'model': config['model'],
        'EP': config['EP'],
        'TP': config['TP'],
        'PP': config['PP'],
        'DP': config['DP'],
        'num_gpus': num_gpus,
        'trace_duration_s': total_duration_s,
    }


    kernel_counts = all_kernels.groupby(['category', 'operation_type']).size().reset_index(name='count')
    kernel_counts = kernel_counts.sort_values('count', ascending=False)

    # for _, row in kernel_counts.iterrows():
    #     pct = (row['count'] / len(all_kernels)) * 100
    #     print(f"{row['category']:15s} - {row['operation_type']:25s}: {row['count']:8,} ({pct:5.2f}%)")

    # ========================================================================
    # COMMUNICATION OPERATIONS
    # ========================================================================

    comm_kernels = all_kernels[all_kernels['category'] == 'Communication']
    compute_kernels = all_kernels[all_kernels['category'] == 'Compute']

    total_comm_time_s = comm_kernels['duration_ns'].sum() / 1e9
    total_compute_time_s = compute_kernels['duration_ns'].sum() / 1e9

    comm_to_compute_ratio = total_comm_time_s / total_compute_time_s if total_compute_time_s > 0 else 0

    results['total_comm_time_s'] = total_comm_time_s
    results['total_compute_time_s'] = total_compute_time_s
    results['comm_to_compute_ratio'] = comm_to_compute_ratio
    results['comm_operations_count'] = len(comm_kernels)
    results['comm_ops_per_second'] = len(comm_kernels) / total_duration_s

    for op_type in comm_kernels['operation_type'].unique():
        op_data = comm_kernels[comm_kernels['operation_type'] == op_type]
        op_time = op_data['duration_ns'].sum() / 1e9
        op_pct = (op_time / total_comm_time_s) * 100 if total_comm_time_s > 0 else 0
        results[f'{op_type}_count'] = len(op_data)
        results[f'{op_type}_time_s'] = op_time

    return float(results["total_compute_time_s"])

def metric_cal(directory: str) -> Dict[str, float]:
    """
    Calculate the kernel compute time from the exported sqlite file from nsys.

    Args:
        directory (str): The directory path containing the exported sqlite file from nsys.

    Returns:
        float: The kernel compute time in seconds. If there are multiple nodes, the compute time from each node is summed.
    """
    dir_name = Path(directory).name
    db_path = str(Path(directory) / "nsys_0.sqlite")
    workload_card_path = Path(directory) / (dir_name + ".yaml")

    # Parse workload card to get metadata
    with open(workload_card_path, 'r') as f:
        workload_card = yaml.safe_load(f)
        model_family = workload_card["workload"]["model"]["model_family"]
        pp = workload_card["Model-executor"]["model_plan_parallelization"]["pp"]

    if model_family not in ["deepseek-v2-lite", "llama-3.1-8B", "qwen-32b"]:
        return "n/a"

    kernel_compute_time = analyze_trace_comprehensive(str(db_path))

    if model_family == "qwen-32b" and pp == 2:
        db_path_1 = str(Path(directory) / "nsys_1.sqlite")
        kernel_compute_time_1 = analyze_trace_comprehensive(db_path_1)
        kernel_compute_time += kernel_compute_time_1
        
    return kernel_compute_time