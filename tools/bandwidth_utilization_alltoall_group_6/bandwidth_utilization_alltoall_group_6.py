import yaml
from pathlib import Path
import pandas as pd
import duckdb

def _get_kernels(con, start_time, end_time, pattern):
    return con.sql(f"""
        SELECT 
            k."start", 
            k."end", 
            k.deviceId, 
            s.value as kernel_name
        FROM sqlite_db.CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN sqlite_db.StringIds s ON k.shortName = s.id
        WHERE k."start" >= {start_time} AND k."end" <= {end_time}
        AND (
            CAST(s.value AS VARCHAR) LIKE '{pattern}' 
        )
    """)


def metric_cal(directory: str) -> float:
    """
    Calculate the bandwidth utilization for alltoall from the exported sqlite file from nsys.

    Only applicable for deepseek. not applicable for ep=1

    Args:
        directory (str): The directory path containing the exported sqlite file from nsys.

    Returns:
        float: The average of non-zero value of NVLink TX Responses User Data [Throughput %], or float("nan") if the metric is not applicable.
    """
    dir_name = Path(directory).name
    db_path = str(Path(directory) / "nsys_0.sqlite")
    workload_card_path = Path(directory) / (dir_name + ".yaml")
    output_csv_path = Path(directory) / "bandwidth_utilization_alltoall_0.csv"

    # Parse workload card to get metadata
    with open(workload_card_path, 'r') as f:
        workload_card = yaml.safe_load(f)
        model_family = workload_card["workload"]["model"]["model_family"]
        ep = workload_card["Model-executor"]["model_plan_parallelization"].get("ep")

    if model_family != "deepseek-v2-lite":
        return float("nan")

    if ep == 1:
        return float("nan")

    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{db_path}' AS sqlite_db (TYPE SQLITE, READ_ONLY);")

    # 2. Get NVTX range (DuckDB is much faster at MIN/MAX on large tables)
    nvtx_range = con.sql("""
        SELECT MIN("start") as s, MAX("end") as e FROM sqlite_db.NVTX_EVENTS
    """).fetchone()

    start_time, end_time = nvtx_range
    splitKReduce = _get_kernels(con, start_time, end_time, "%splitKreduce%")
    moesumreduce = _get_kernels(con, start_time, end_time, "%moe_sum_reduce_warp%")
    
    events_df = con.sql("""
    SELECT 
        s."start" AS "start",
        m."end" AS "end",
        m.deviceId
    FROM moesumreduce m
    ASOF JOIN splitKReduce s
        ON m.deviceId = s.deviceId
        AND m."start" >= s."end"
    """).df()

    # 5. Extract Metric IDs
    all_metrics = con.sql("""
        SELECT DISTINCT metricId, 
        FROM sqlite_db.TARGET_INFO_GPU_METRICS 
        WHERE metricName LIKE '%NVLink%'
    """).df()
    raw_stats = con.sql("""
    SELECT 
        m.metricID,
        MIN(m.value) AS min_val,
        MAX(m.value) AS max_val,
        AVG(m.value) AS avg_val,
        MIN(CASE WHEN m.value != 0 THEN m.value END) AS min_no_zero,
        AVG(CASE WHEN m.value != 0 THEN m.value END) AS avg_no_zero,
        COUNT(CASE WHEN m.value != 0 THEN m.value END) as cnt_no_zero,
        MIN(CASE WHEN m.value > 1 THEN m.value END) AS min_gt_one,
        AVG(CASE WHEN m.value > 1 THEN m.value END) AS avg_gt_one, 
        COUNT(CASE WHEN m.value > 1 THEN m.value END) as cnt_gt_one
    FROM sqlite_db.GPU_METRICS m
    INNER JOIN events_df e 
        ON (m.typeId & 255) = e.deviceID
        AND m.timestamp >= e.start 
        AND m.timestamp <= e.end
    INNER JOIN all_metrics f
        ON m.metricID = f.metricID
    GROUP BY m.metricID
    """).df()

    metric_mapping = con.sql("""
    SELECT DISTINCT 
        metricId, 
        metricName 
    FROM sqlite_db.TARGET_INFO_GPU_METRICS 
    WHERE metricName LIKE '%NVLink%'
    """)

    results_df = con.sql("""
        SELECT 
            f.metricName,
            r.*
        FROM raw_stats r
        LEFT JOIN metric_mapping f ON r.metricID = f.metricId
        ORDER BY f.metricName
    """).df()

    results_df.to_csv(output_csv_path)

    row = results_df.loc[results_df["metricName"] == "NVLink TX Responses User Data [Throughput %]"]

    if row.empty:
        return float("nan")

    row = row.iloc[0]

    ret = float(row["avg_no_zero"])

    return ret