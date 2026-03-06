#!/usr/bin/env python3
"""
Plot NVLink throughput over time for a specific link and direction.

Usage:
  python plot_throughput.py /path/to/trace_dir --link 0 --direction tx
  python plot_throughput.py /path/to/trace_dir --link 0 --direction rx --output /path/to/output
  python plot_throughput.py /path/to/trace_dir --link all --direction tx  # Plot all links
"""

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


# NVLink trace format
FILE_MAGIC = b"NVF1"
HEADER_STRUCT = struct.Struct("<4sHHi")

# nvmlFieldValue_t offsets
NVML_FIELD_ID_OFFSET = 0
NVML_SCOPE_ID_OFFSET = 4
NVML_TIMESTAMP_OFFSET = 8
NVML_VALUE_OFFSET = 32

# NVML field IDs
NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX = 138
NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX = 139


def field_id_to_direction(field_id: int) -> str:
    if field_id == NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX:
        return "TX"
    elif field_id == NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX:
        return "RX"
    else:
        return f"F{field_id}"


@dataclass
class NVLinkSample:
    nvml_timestamp_us: int
    field_id: int
    link_id: int
    value: int


@dataclass
class NVLinkInterval:
    start_us: int
    end_us: int
    field_id: int
    link_id: int
    delta_bytes: int

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us

    @property
    def direction(self) -> str:
        return field_id_to_direction(self.field_id)

    @property
    def delta_bytes_actual(self) -> int:
        return self.delta_bytes * 1024

    @property
    def throughput_gbps(self) -> float:
        if self.duration_us <= 0:
            return 0.0
        bytes_transferred = self.delta_bytes * 1024
        bytes_per_sec = bytes_transferred * 1e6 / self.duration_us
        return bytes_per_sec / 1e9

    @property
    def midpoint_us(self) -> float:
        return (self.start_us + self.end_us) / 2


def load_nvlink_trace(trace_file: Path) -> List[NVLinkSample]:
    """Load NVLink utilization trace from binary file."""
    samples = []

    with open(trace_file, "rb") as f:
        header_data = f.read(HEADER_STRUCT.size)
        magic, version, stored_field_size, host_ts_size = HEADER_STRUCT.unpack(header_data)

        if magic != FILE_MAGIC:
            raise ValueError(f"Invalid magic: {magic}")

        record_stride = host_ts_size + stored_field_size
        while True:
            record = f.read(record_stride)
            if len(record) < record_stride:
                break

            field_data = record[host_ts_size:]
            field_id = struct.unpack("<I", field_data[NVML_FIELD_ID_OFFSET:NVML_FIELD_ID_OFFSET+4])[0]
            scope_id = struct.unpack("<I", field_data[NVML_SCOPE_ID_OFFSET:NVML_SCOPE_ID_OFFSET+4])[0]
            nvml_ts_us = struct.unpack("<Q", field_data[NVML_TIMESTAMP_OFFSET:NVML_TIMESTAMP_OFFSET+8])[0]
            value = struct.unpack("<Q", field_data[NVML_VALUE_OFFSET:NVML_VALUE_OFFSET+8])[0]

            samples.append(NVLinkSample(
                nvml_timestamp_us=nvml_ts_us,
                field_id=field_id,
                link_id=scope_id,
                value=value,
            ))

    return samples


def compute_nvlink_intervals(samples: List[NVLinkSample]) -> List[NVLinkInterval]:
    """Compute throughput intervals between consecutive samples."""
    by_key: Dict[Tuple[int, int], List[NVLinkSample]] = {}
    for s in samples:
        key = (s.field_id, s.link_id)
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(s)

    intervals = []
    for key, key_samples in by_key.items():
        key_samples.sort(key=lambda x: x.nvml_timestamp_us)

        for i in range(1, len(key_samples)):
            prev = key_samples[i-1]
            curr = key_samples[i]
            delta = curr.value - prev.value
            if delta < 0:
                delta = 0

            intervals.append(NVLinkInterval(
                start_us=prev.nvml_timestamp_us,
                end_us=curr.nvml_timestamp_us,
                field_id=curr.field_id,
                link_id=curr.link_id,
                delta_bytes=delta,
            ))

    return intervals


def filter_intervals(
    intervals: List[NVLinkInterval],
    link: Optional[int] = None,
    direction: Optional[str] = None
) -> List[NVLinkInterval]:
    """Filter intervals by link and/or direction."""
    result = intervals

    if link is not None:
        result = [iv for iv in result if iv.link_id == link]

    if direction is not None:
        dir_upper = direction.upper()
        result = [iv for iv in result if iv.direction == dir_upper]

    return result


def plot_throughput(
    trace_dir: str,
    link: Optional[int] = None,
    direction: str = "TX",
    output_dir: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    log_scale: bool = True,
):
    """
    Plot throughput over time for specified link(s) and direction.
    
    Args:
        trace_dir: Path to trace directory containing *.bin trace file
        link: Link ID to plot (None = plot all links on same graph)
        direction: "TX" or "RX"
        output_dir: Directory to save plot (if None, uses trace_dir)
        show: Whether to display the plot interactively
        title: Custom title for the plot
        figsize: Figure size (width, height) in inches
        log_scale: Use logarithmic scale for y-axis (default: True)
    """
    trace_path = Path(trace_dir)
    bin_files = list(trace_path.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No .bin trace file found in: {trace_path}")
    nvlink_file = bin_files[0]  # Use first .bin file found
    
    # Load and compute intervals
    samples = load_nvlink_trace(nvlink_file)
    intervals = compute_nvlink_intervals(samples)
    
    # Filter by direction
    direction = direction.upper()
    intervals = [iv for iv in intervals if iv.direction == direction and iv.duration_us > 0]
    
    if not intervals:
        print(f"No intervals found for direction {direction}")
        return
    
    # Get available links
    available_links = sorted(set(iv.link_id for iv in intervals))
    print(f"Available links for {direction}: {available_links}")
    
    # Determine which links to plot
    if link is not None:
        links_to_plot = [link]
        if link not in available_links:
            print(f"Warning: Link {link} not found in trace")
            return
    else:
        links_to_plot = available_links
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(links_to_plot)))
    
    # Find global time range for normalization
    all_times = [iv.midpoint_us for iv in intervals]
    min_time = min(all_times)
    
    # Plot each link
    for idx, link_id in enumerate(links_to_plot):
        link_intervals = [iv for iv in intervals if iv.link_id == link_id]
        link_intervals.sort(key=lambda x: x.start_us)
        
        # Convert to relative time (seconds from start)
        times_sec = [(iv.midpoint_us - min_time) / 1e6 for iv in link_intervals]
        throughputs = [iv.throughput_gbps for iv in link_intervals]
        
        label = f"Link {link_id}" if len(links_to_plot) > 1 else f"Link {link_id} {direction}"
        ax.plot(times_sec, throughputs, '-', color=colors[idx], label=label, 
                linewidth=0.8, alpha=0.8)
    
    # Labels and title
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ylabel = "Throughput (GB/s)" + (" [log scale]" if log_scale else "")
    ax.set_ylabel(ylabel, fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        trace_name = trace_path.name
        link_str = f"Link {link}" if link is not None else "All Links"
        ax.set_title(f"NVLink {direction} Throughput - {link_str}\n{trace_name}", fontsize=14)
    
    # Legend
    if len(links_to_plot) > 1:
        ax.legend(loc='upper right', fontsize=10)
    
    # Log scale
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-4)  # Set minimum to avoid log(0)
    else:
        ax.set_ylim(bottom=0)
    
    # Grid
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(left=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = trace_path
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    link_suffix = f"link{link}" if link is not None else "all_links"
    scale_suffix = "_log" if log_scale else ""
    filename = f"nvlink_throughput_{direction.lower()}_{link_suffix}{scale_suffix}.png"
    output_file = output_path / filename
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_file


def plot_throughput_combined(
    trace_dir: str,
    link: Optional[int] = None,
    output_dir: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (14, 10),
    log_scale: bool = True,
):
    """
    Plot both TX and RX throughput in subplots.
    """
    trace_path = Path(trace_dir)
    bin_files = list(trace_path.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No .bin trace file found in: {trace_path}")
    nvlink_file = bin_files[0]  # Use first .bin file found
    
    # Load and compute intervals
    samples = load_nvlink_trace(nvlink_file)
    intervals = compute_nvlink_intervals(samples)
    intervals = [iv for iv in intervals if iv.duration_us > 0]
    
    if not intervals:
        print("No intervals found")
        return
    
    # Filter by link if specified
    if link is not None:
        intervals = [iv for iv in intervals if iv.link_id == link]
    
    # Separate TX and RX
    tx_intervals = [iv for iv in intervals if iv.direction == "TX"]
    rx_intervals = [iv for iv in intervals if iv.direction == "RX"]
    
    # Get available links
    tx_links = sorted(set(iv.link_id for iv in tx_intervals))
    rx_links = sorted(set(iv.link_id for iv in rx_intervals))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Find global time range
    all_times = [iv.midpoint_us for iv in intervals]
    min_time = min(all_times)
    
    # Plot TX
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(tx_links), len(rx_links))))
    
    for idx, link_id in enumerate(tx_links):
        link_ivs = [iv for iv in tx_intervals if iv.link_id == link_id]
        link_ivs.sort(key=lambda x: x.start_us)
        times_sec = [(iv.midpoint_us - min_time) / 1e6 for iv in link_ivs]
        throughputs = [iv.throughput_gbps for iv in link_ivs]
        ax1.plot(times_sec, throughputs, '-', color=colors[idx], 
                label=f"Link {link_id}", linewidth=0.8, alpha=0.8)
    
    ylabel_suffix = " [log]" if log_scale else ""
    ax1.set_ylabel(f"TX Throughput (GB/s){ylabel_suffix}", fontsize=12)
    ax1.set_title(f"NVLink TX Throughput", fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')
    if log_scale:
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=1e-4)
    else:
        ax1.set_ylim(bottom=0)
    if len(tx_links) > 1:
        ax1.legend(loc='upper right', fontsize=9)
    
    # Plot RX
    for idx, link_id in enumerate(rx_links):
        link_ivs = [iv for iv in rx_intervals if iv.link_id == link_id]
        link_ivs.sort(key=lambda x: x.start_us)
        times_sec = [(iv.midpoint_us - min_time) / 1e6 for iv in link_ivs]
        throughputs = [iv.throughput_gbps for iv in link_ivs]
        ax2.plot(times_sec, throughputs, '-', color=colors[idx], 
                label=f"Link {link_id}", linewidth=0.8, alpha=0.8)
    
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel(f"RX Throughput (GB/s){ylabel_suffix}", fontsize=12)
    ax2.set_title(f"NVLink RX Throughput", fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(left=0)
    if log_scale:
        ax2.set_yscale('log')
        ax2.set_ylim(bottom=1e-4)
    else:
        ax2.set_ylim(bottom=0)
    if len(rx_links) > 1:
        ax2.legend(loc='upper right', fontsize=9)
    
    # Overall title
    trace_name = trace_path.name
    link_str = f"Link {link}" if link is not None else "All Links"
    fig.suptitle(f"NVLink Throughput - {link_str}\n{trace_name}", fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = trace_path
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    link_suffix = f"link{link}" if link is not None else "all_links"
    scale_suffix = "_log" if log_scale else ""
    filename = f"nvlink_throughput_combined_{link_suffix}{scale_suffix}.png"
    output_file = output_path / filename
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Plot NVLink throughput over time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot TX throughput for link 0 (log scale by default)
  %(prog)s /path/to/trace_dir --link 0 --direction tx

  # Plot RX throughput for all links
  %(prog)s /path/to/trace_dir --direction rx

  # Plot both TX and RX in combined plot
  %(prog)s /path/to/trace_dir --combined

  # Use linear scale instead of log
  %(prog)s /path/to/trace_dir --link 0 --direction tx --linear

  # Save to specific output directory
  %(prog)s /path/to/trace_dir --link 0 --direction tx --output /path/to/plots

  # Show plot interactively
  %(prog)s /path/to/trace_dir --link 0 --direction tx --show
"""
    )
    parser.add_argument("trace_dir", type=str,
                        help="Directory containing *.bin trace file")
    parser.add_argument("--link", "-l", type=int, default=None,
                        help="Link ID to plot (default: all links)")
    parser.add_argument("--direction", "-d", type=str,
                        choices=["tx", "rx", "TX", "RX"],
                        default="TX",
                        help="Direction to plot (default: TX)")
    parser.add_argument("--combined", "-c", action="store_true",
                        help="Plot both TX and RX in combined figure")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for plot (default: trace_dir)")
    parser.add_argument("--show", "-s", action="store_true",
                        help="Show plot interactively")
    parser.add_argument("--title", "-t", type=str, default=None,
                        help="Custom title for the plot")
    parser.add_argument("--width", type=int, default=14,
                        help="Figure width in inches (default: 14)")
    parser.add_argument("--height", type=int, default=6,
                        help="Figure height in inches (default: 6)")
    parser.add_argument("--linear", action="store_true",
                        help="Use linear scale instead of log scale for y-axis")

    args = parser.parse_args()
    
    log_scale = not args.linear

    try:
        if args.combined:
            plot_throughput_combined(
                args.trace_dir,
                link=args.link,
                output_dir=args.output,
                show=args.show,
                figsize=(args.width, args.height * 2),
                log_scale=log_scale,
            )
        else:
            plot_throughput(
                args.trace_dir,
                link=args.link,
                direction=args.direction,
                output_dir=args.output,
                show=args.show,
                title=args.title,
                figsize=(args.width, args.height),
                log_scale=log_scale,
            )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        exit(1)
    except ImportError as e:
        print(f"ERROR: Missing dependency - {e}")
        print("Install with: pip install matplotlib numpy")
        exit(1)


if __name__ == "__main__":
    main()

