# NVLink Usage Analysis

Analyze NVLink throughput metrics from binary trace files.

## Metrics

| Metric | Description |
|--------|-------------|
| `max_throughput` | Maximum throughput (GB/s) with link, direction, and timestamp |
| `avg_throughput` | Average throughput (GB/s) weighted by interval duration |
| `total_communication` | Total bytes transferred (RX+TX)/2 for symmetry |

## Usage

### Command Line

```bash
# Calculate all metrics
python analyze_nvlink_throughput.py /path/to/trace_dir --metrics all

# Calculate specific metrics
python analyze_nvlink_throughput.py /path/to/trace_dir --metrics max_throughput avg_throughput

# Filter by link and direction
python analyze_nvlink_throughput.py /path/to/trace_dir --metrics all --link 0 --direction tx

# Output as JSON
python analyze_nvlink_throughput.py /path/to/trace_dir --metrics all --json
```

### Via main.py

```bash
python main.py --trace /path/to/trace_dir --metric max_throughput
python main.py --trace /path/to/trace_dir --metric avg_throughput --link 0
python main.py --trace /path/to/trace_dir --metric nvlink_all --json
```

### Direct Import

```python
from nvlink_usage.analyze_nvlink_throughput import metric_cal

# Get all metrics
results = metric_cal("/path/to/trace_dir", metric_name="all")

# Get specific metric with filters
results = metric_cal("/path/to/trace_dir", 
                     metric_name="max_throughput",
                     link=0, 
                     direction="tx")
```

## Input Requirements

The trace directory must contain:
- `*.bin` - Binary NVLink trace file (from NVML sampling)

## Output Format

### max_throughput
```json
{
  "max_throughput_gbps": 1.7337,
  "link_id": 0,
  "direction": "RX",
  "timestamp_us": 1766219131558214,
  "interval_duration_us": 1234,
  "interval_bytes": 2134016
}
```

### avg_throughput
```json
{
  "avg_throughput_gbps": 0.0065,
  "total_duration_us": 300000000,
  "num_intervals": 5000,
  "num_active_intervals": 1234
}
```

### total_communication
```json
{
  "total_bytes": 1234567890,
  "total_gb": 1.2346,
  "rx_bytes": 1234567890,
  "tx_bytes": 1234567890
}
```

