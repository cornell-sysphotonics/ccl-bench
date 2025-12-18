import yaml
from pathlib import Path
import sqlite3
import pandas as pd


def _get_dtod_memcpy_for_device(conn, device_id: int) -> pd.DataFrame:
    dtod_copykind_value: int = 8
    query = f"""
    SELECT 
        start, 
        end, 
        bytes,
        deviceId,
        correlationId, 
        globalPid, 
        srcDeviceId, 
        dstDeviceId, 
        copyKind,
        streamId
    FROM 
        CUPTI_ACTIVITY_KIND_MEMCPY
    WHERE 
        -- Filter 1: The device must be involved (either source or destination)
        deviceId = {device_id}
        AND copyKind={dtod_copykind_value}
    """
    memcpy_df = pd.read_sql_query(query, conn)
    return memcpy_df

def _get_reduce_kernels_for_device(conn, device_id: int, kernel_name_prefix: str = '%cross_device_reduce__stage%') -> pd.DataFrame:
    query = f"""
    SELECT 
        T1.start, 
        T1.end, 
        T1.deviceId, 
        T1.streamId, 
        T1.correlationId,
        T2.value AS KernelName
    FROM 
        CUPTI_ACTIVITY_KIND_KERNEL T1
    INNER JOIN 
        StringIds T2 ON T1.shortName = T2.id
    WHERE 
        -- Filter 1: Match the specific GPU device
        T1.deviceId = {device_id}
        -- Filter 2: Match the kernel name prefix using LIKE
        AND T2.value LIKE '{kernel_name_prefix}'
    ORDER BY
        T1.start
    """
    kernel_df = pd.read_sql_query(query, conn)
    return kernel_df

def _find_reduce_pattern(
        memcpy_df: pd.DataFrame, 
        kernel_df: pd.DataFrame, 
        max_delay_ns = 20000
) -> pd.DataFrame:

    memcpy_prepared = memcpy_df[['end', 'start', 'bytes', 'correlationId','deviceId']].copy()
    memcpy_prepared = memcpy_prepared.rename(columns={
        'end': 'memcpy_end',
        'start': 'memcpy_start',
        'correlationId': 'memcpy_correlationId'
    })
    memcpy_prepared = memcpy_prepared.sort_values('memcpy_end')

    # Prepare kernel dataframe
    kernel_prepared = kernel_df[['start', 'end', 'KernelName', 'correlationId']].copy()
    kernel_prepared = kernel_prepared.rename(columns={
        'start': 'kernel_start',
        'end': 'kernel_end',
        'correlationId': 'kernel_correlationId'
    })
    kernel_prepared = kernel_prepared.sort_values('kernel_start')

    # Use merge_asof to find the first kernel that starts after each memcpy ends
    # direction='forward' means find the next kernel after memcpy
    # tolerance=20000 means within 20,000 ns
    results_df = pd.merge_asof(
        memcpy_prepared,
        kernel_prepared,
        left_on='memcpy_end',
        right_on='kernel_start',
        direction='forward',
        tolerance=max_delay_ns
    )

    return results_df.dropna()

def _get_bandwidth_utilization(combined_df, NGPUS=4, bandwidth=600): 
    combined_df['kernel duration(s)'] = (combined_df['kernel_end'] - combined_df['kernel_start'])/(10**9)
    # reduce is 2*(N-1)/N
    # we cannot join from different slices, but we will assume that each slice has the same number
    # we calculate for each gpu indiviudally and presumably, final outputs should have 4x of each transfer 
    combined_df['effective bandwidth(GB/s)'] = (combined_df['bytes'] * 4) / (2**30) * (2 * (NGPUS-1)/(NGPUS)) / combined_df['kernel duration(s)']
    combined_df['bandwidth utilization'] = combined_df['effective bandwidth(GB/s)']/bandwidth
    return combined_df

def metric_cal(directory: str) -> float:
    """
    Calculate the bandwidth utilization for allreduce from the exported sqlite file from nsys.

    n/a for llama tp=1

    Args:
        directory (str): The directory path containing the exported sqlite file from nsys.

    Returns:
        Dict[str, float] | "n/a": The statistics of bandwidth utilization for allreduce, or "n/a" if the metric is not applicable.
    """
    dir_name = Path(directory).name
    db_path = str(Path(directory) / "nsys_0.sqlite")
    workload_card_path = Path(directory) / (dir_name + ".yaml")
    output_csv_path = Path(directory) / "bandwidth_utilization_allreduce.csv"

    # Parse workload card to get metadata
    with open(workload_card_path, 'r') as f:
        workload_card = yaml.safe_load(f)
        model_family = workload_card["workload"]["model"]["model_family"]
        tp = workload_card["Model-executor"]["model_plan_parallelization"]["tp"]

    if model_family == "llama-3.1-8B" and tp == 1:
        return "n/a"

    try: 
        conn = sqlite3.connect(db_path)
        df_list=[]
        for i in range(4): 
            memcpy_df = _get_dtod_memcpy_for_device(conn, i)
            reduce_kernels = _get_reduce_kernels_for_device(conn, i)
            df = _find_reduce_pattern(memcpy_df, reduce_kernels)
            df_list.append(df.copy())
        combined_df = pd.concat(df_list, axis=0, ignore_index=True)
        bandwidth_utilization = _get_bandwidth_utilization(combined_df)
        bandwidth_utilization.to_csv(output_csv_path)
        # print(f'saved to {output_csv_path}')
        # print(bandwidth_utilization.describe())
    except Exception as e: 
        print('error in querying', e)
        return "n/a"

    col = bandwidth_utilization["bandwidth utilization"]

    stats = {
        "mean": float(col.mean()),
        "median": float(col.median()),
        "std": float(col.std()),
        "p25": float(col.quantile(0.25)),
        "p75": float(col.quantile(0.75)),
        "p99": float(col.quantile(0.99)),
    }

    return stats