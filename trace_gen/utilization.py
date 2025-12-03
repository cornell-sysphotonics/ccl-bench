#!/usr/bin/env python3
import argparse
import queue
import signal
import struct
import threading
import time
from ctypes import addressof, sizeof, string_at
from pathlib import Path

from pynvml import *

# Number of NVLINKs wired per GPU on the target system (A100 has 12).
NVLINKS_PER_GPU = 12

# Emit (fieldId, scopeId) for TX/RX counters across all NVLINKs on the GPU.
POLL_TARGETS = [
    (field, link_id)
    for link_id in range(NVLINKS_PER_GPU)
    for field in (
        NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX,
        NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX,
    )
]

FILE_MAGIC = b"NVF1"
HEADER_STRUCT = struct.Struct("<4sHHi")          # magic, version, field_size, host_ts_size
HOST_TS_STRUCT = struct.Struct("<Q")             # perf_counter_ns
FIELD_SIZE = sizeof(c_nvmlFieldValue_t)
RECORD_STRIDE = HOST_TS_STRUCT.size + FIELD_SIZE


def writer_worker(out_path: Path, q: queue.Queue, stop_evt: threading.Event) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb", buffering=1024 * 1024) as fh:
        fh.write(HEADER_STRUCT.pack(FILE_MAGIC, 1, FIELD_SIZE, HOST_TS_STRUCT.size))
        while True:
            if stop_evt.is_set() and q.empty():
                break
            try:
                host_ts, raw = q.get(timeout=0.05)
            except queue.Empty:
                continue
            fh.write(HOST_TS_STRUCT.pack(host_ts))
            fh.write(raw)


def enqueue(q: queue.Queue, item, stop_evt: threading.Event) -> None:
    while not stop_evt.is_set():
        try:
            q.put(item, timeout=0.05)
            return
        except queue.Full:
            continue


def poll_loop(handle, requests, interval_s, q: queue.Queue, stop_evt: threading.Event):
    next_tick = time.perf_counter()
    while not stop_evt.is_set():
        next_tick += interval_s
        delay = next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        values = nvmlDeviceGetFieldValues(handle, requests)
        host_ts = time.perf_counter_ns()
        for val in values:
            raw = string_at(addressof(val), FIELD_SIZE)
            enqueue(q, (host_ts, raw), stop_evt)


def main():
    parser = argparse.ArgumentParser(description="NVML field poller with background writer.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to poll.")
    parser.add_argument("--interval-ms", type=float, default=1.0,
                        help="Polling period in milliseconds (host-side cadence).")
    parser.add_argument("--out", type=Path, default=Path("nvlink_trace.bin"),
                        help="Binary output file for captured field values.")
    parser.add_argument("--queue-depth", type=int, default=8192,
                        help="Max records buffered between poller and writer.")
    args = parser.parse_args()

    interval_s = args.interval_ms / 1000.0
    stop_evt = threading.Event()
    record_queue: queue.Queue = queue.Queue(maxsize=args.queue_depth)

    def handle_signal(signum, _frame):
        print(f"\nSignal {signum} received, stopping…")
        stop_evt.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_signal)

    nvmlInit()
    try:
        handle = nvmlDeviceGetHandleByIndex(args.gpu)
        requests = list(POLL_TARGETS)

        writer = threading.Thread(
            target=writer_worker,
            args=(args.out.resolve(), record_queue, stop_evt),
            name="nvml-writer",
            daemon=True,
        )
        writer.start()

        try:
            poll_loop(handle, requests, interval_s, record_queue, stop_evt)
        finally:
            stop_evt.set()
            writer.join()
    finally:
        nvmlShutdown()


if __name__ == "__main__":
    main()