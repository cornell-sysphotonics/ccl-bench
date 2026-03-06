# NVLink Plotting

Generate throughput-over-time plots for NVLink traces.

## Usage

### Command Line

```bash
# Plot TX throughput for link 0 (log scale by default)
python plot_throughput.py /path/to/trace_dir --link 0 --direction tx

# Plot RX throughput for all links
python plot_throughput.py /path/to/trace_dir --direction rx

# Plot both TX and RX in combined subplots
python plot_throughput.py /path/to/trace_dir --combined

# Use linear scale instead of log
python plot_throughput.py /path/to/trace_dir --link 0 --direction tx --linear

# Save to specific output directory
python plot_throughput.py /path/to/trace_dir --link 0 --direction tx --output /path/to/plots

# Show plot interactively
python plot_throughput.py /path/to/trace_dir --link 0 --direction tx --show

# Custom title and size
python plot_throughput.py /path/to/trace_dir --link 0 -d tx --title "My Plot" --width 16 --height 8
```

## Options

| Option | Description |
|--------|-------------|
| `--link`, `-l` | Link ID to plot (default: all links) |
| `--direction`, `-d` | Direction: `tx` or `rx` (default: TX) |
| `--combined`, `-c` | Plot both TX and RX in combined figure |
| `--output`, `-o` | Output directory for plot (default: trace_dir) |
| `--show`, `-s` | Show plot interactively |
| `--title`, `-t` | Custom title for the plot |
| `--linear` | Use linear scale instead of log scale |
| `--width` | Figure width in inches (default: 14) |
| `--height` | Figure height in inches (default: 6) |

## Input Requirements

The trace directory must contain:
- `*.bin` - Binary NVLink trace file

## Output

Generates PNG files:
- Single direction: `nvlink_throughput_tx_link0_log.png`
- All links: `nvlink_throughput_rx_all_links_log.png`
- Combined: `nvlink_throughput_combined_link0_log.png`
- Linear scale: `nvlink_throughput_tx_link0.png` (no `_log` suffix)

## Dependencies

```bash
pip install matplotlib numpy
```

