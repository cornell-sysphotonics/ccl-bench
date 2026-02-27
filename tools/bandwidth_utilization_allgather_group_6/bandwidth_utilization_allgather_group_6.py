"""
This script collects information about all gather nccl calls. Specifically, when tp>1, we see that there is an all gather at the end of a model call that corresponds to the final logits of the output. 
- we set the side of the buffer to be equal to the vocab size (as that will be the amount of data transfered)
- we note that the bandwidth utilization is low --> as the logits are very small (MBs), we likely are spending most of our time in set up and overhead rather than actually transfering data 
- this script will output information about nvtx evetns, kernel launch and running information, as well as our final calculations for bandwidth utilization 
"""

import yaml
from pathlib import Path
import sqlite3
import pandas as pd

def _get_nvtx_events(conn, pattern="%logits_processor%"): 
    nvtx_query = f"""
    SELECT start, end, globalTid, domainId, text, eventType
    FROM NVTX_EVENTS
    WHERE text LIKE '{pattern}';
    """
    nvtx_events = pd.read_sql_query(nvtx_query, conn)
    return nvtx_events

def _get_cuda_kernel(conn, pattern="%ncclDevKernel_AllGather_RING_LL%"):
    cuda_kernel_query = f"""
    SELECT k.start, k.end, k.deviceId as gpu_id, k.correlationId, n.value as kernelName
    FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
    JOIN StringIds AS n ON k.demangledName = n.id
    WHERE n.value LIKE '{pattern}';
    """
    cuda_kernels = pd.read_sql_query(cuda_kernel_query, conn)
    return cuda_kernels


def _get_cuda_api(conn, pattern="%cu%Launch%"):
    cuda_api_query = f"""
    SELECT c.start, c.end, c.globalTid, c.correlationId, s.value as name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME as c
    JOIN StringIds AS s ON c.nameId = s.id 
    WHERE s.value LIKE '{pattern}'; 
    """
    cuda_apis = pd.read_sql_query(cuda_api_query, conn)
    return cuda_apis

def _merge_dfs(nvtx_events, cuda_kernels, cuda_apis): 
    
    correlated_data = []

    for _, nvtx_row in nvtx_events.iterrows():
        # Filter API calls that are both temporally and contextually within the NVTX range
        matching_apis = cuda_apis[
            (cuda_apis['start'] >= nvtx_row['start']) &
            (cuda_apis['end'] <= nvtx_row['end']) &
            (cuda_apis['globalTid'] == nvtx_row['globalTid'])
        ].copy() # Use copy to avoid SettingWithCopyWarning when adding columns

        if not matching_apis.empty:
            matching_apis['nvtx_name'] = nvtx_row['text']
            # Store these matched API calls to link to kernels next
            correlated_data.append(matching_apis)

    # Combine all matched API calls into a single DataFrame
    if correlated_data:
        df_matched_apis = pd.concat(correlated_data, ignore_index=True)
    else:
        # Handle case where no API calls match NVTX ranges
        df_matched_apis = pd.DataFrame()

    # 2. Second Join: Link the matched API calls to GPU Kernels using correlationId

    final_joined_data = pd.merge(
        df_matched_apis,
        cuda_kernels,
        on='correlationId',
        suffixes=('_api', '_kernel')
    )

    return final_joined_data

def _get_bandwidth_utilization(final_joined_data, vocab_size_bytes, bandwidth=600, NGPUS=4): 
    final_joined_data['duration(ns)'] = final_joined_data['end_kernel'] - final_joined_data['start_kernel']
    final_joined_data['duration(s)'] = final_joined_data['duration(ns)']/(10**9)
    vocab_size_GB = vocab_size_bytes/(2**30)

    final_joined_data['effective bandwidth(GB/S)'] = vocab_size_GB * (1*(NGPUS-1)/NGPUS)  /  final_joined_data['duration(s)']
    final_joined_data['bandwidth utilization'] = final_joined_data['effective bandwidth(GB/S)']/bandwidth
    return final_joined_data


def metric_cal(directory: str) -> float:
    """
    Calculate the bandwidth utilization for allgather from the exported sqlite file from nsys.

    Note that AllGather has only been calculated for tp>1 and for the last stage of pp when pp>1.

    n/a for llama tp=1 and node 0 of qwen pp=2

    Args:
        directory (str): The directory path containing the exported sqlite file from nsys.

    Returns:
        float: The median bandwidth utilization for allgather, or float("nan") if the metric is not applicable.
    """
    dir_name = Path(directory).name
    db_path = str(Path(directory) / "nsys_0.sqlite")
    workload_card_path = Path(directory) / (dir_name + ".yaml")
    output_csv_path = Path(directory) / "bandwidth_utilization_allgather_0.csv"

    # Parse workload card to get metadata
    with open(workload_card_path, 'r') as f:
        workload_card = yaml.safe_load(f)
        model_family = workload_card["workload"]["model"]["model_family"]
        tp = workload_card["Model-executor"]["model_plan_parallelization"]["tp"]
        pp = workload_card["Model-executor"]["model_plan_parallelization"]["pp"]

    if model_family not in ["deepseek-v2-lite", "llama-3.1-8B", "qwen-32b"]:
        return float("nan")

    # vocab sizes are found from model's config.json
    if model_family == "deepseek-v2-lite":
        vocab_size=102400*2
    elif model_family == "llama-3.1-8B":
        if tp == 1:
            return float("nan")
        vocab_size=128256*2
    elif model_family == "qwen-32b":
        vocab_size=151936*2
        # allgather is only applicable for node 1 for qwen pp=2
        if pp == 2:
            db_path = str(Path(directory) / "nsys_1.sqlite")
            output_csv_path = Path(directory) / "bandwidth_utilization_allgather_1.csv"
    else:
        return float("nan")

    try: 
        conn = sqlite3.connect(db_path)
        nvtx_events = _get_nvtx_events(conn)
        cuda_kernel = _get_cuda_kernel(conn)
        cuda_api = _get_cuda_api(conn)
        final_joined_df = _merge_dfs(nvtx_events, cuda_kernel, cuda_api)
        bandwidth_utilization = _get_bandwidth_utilization(final_joined_df, vocab_size)
        bandwidth_utilization.to_csv(output_csv_path)
        # print(f'saved to {output_csv_path}')
    except Exception as e: 
        print('error in querying', e)
        return float("nan")

    col = bandwidth_utilization["bandwidth utilization"]

    return float(col.median())