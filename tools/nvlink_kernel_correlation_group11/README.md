# NVLink Kernel Correlation

Correlate NVLink throughput peaks with GPU kernel activity from nsys traces.

## Metrics

| Metric | Description |
|--------|-------------|
| `peak_kernels` | Kernels (especially NCCL) during the highest throughput interval(s) |

## Usage

### Command Line

```bash
# Find kernels during peak throughput (GPU 0, all links)
python correlate_nsys_nvlink.py /path/to/trace_dir --metrics peak_kernels

# Filter by GPU
python correlate_nsys_nvlink.py /path/to/trace_dir --gpu 1

# Filter by link and direction
python correlate_nsys_nvlink.py /path/to/trace_dir --link 0 --direction tx

# Show top 5 peak intervals
python correlate_nsys_nvlink.py /path/to/trace_dir --top 5

# Only show NCCL kernels
python correlate_nsys_nvlink.py /path/to/trace_dir --nccl-only

# Output as JSON
python correlate_nsys_nvlink.py /path/to/trace_dir --json
```

### Via main.py

```bash
python main.py --trace /path/to/trace_dir --metric peak_kernels
python main.py --trace /path/to/trace_dir --metric peak_kernels --gpu 0 --top 3
python main.py --trace /path/to/trace_dir --metric peak_kernels --nccl-only --json
```

### Direct Import

```python
from nvlink_kernel_correlation.correlate_nsys_nvlink import metric_cal

results = metric_cal("/path/to/trace_dir",
                     metric_name="peak_kernels",
                     gpu=0,
                     link=None,      # all links
                     direction=None, # all directions
                     top=1,
                     nccl_only=False)
```

## Input Requirements

The trace directory must contain:
- `*.bin` - Binary NVLink trace file
- `*.sqlite` - nsys SQLite export (e.g., `vllm_profile.sqlite`)

To generate the SQLite file from nsys:
```bash
nsys export --type sqlite --output profile.sqlite profile.nsys-rep
```

## Output Format

```json
{
  "peak_kernels": {
    "peak_interval": {
      "start_us": 1766219131558214,
      "throughput_gbps": 1.7337,
      "link_id": 0,
      "direction": "RX",
      "timestamp_iso": "2025-12-19T23:45:31.558Z",
      "bytes_transferred": 2134016
    },
    "kernels": [
      {
        "name": "ncclKernel_AllReduce_RING_LL_Sum_float",
        "overlap_pct": 95.2,
        "overlap_us": 1175,
        "is_nccl": true,
        "duration_us": 1500,
        "device_id": 0
      }
    ],
    "num_kernels": 15,
    "num_nccl_kernels": 3
  }
}
```

Kernels are sorted by overlap with the peak interval (descending).

