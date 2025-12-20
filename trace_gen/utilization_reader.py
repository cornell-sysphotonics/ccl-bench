#!/usr/bin/env python3
"""Binary reader for NVML field capture files produced by utilization.py."""
import argparse
import struct
from ctypes import sizeof
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore[import]

pd = None
from pynvml import (
    NVML_VALUE_TYPE_DOUBLE,
    NVML_VALUE_TYPE_SIGNED_INT,
    NVML_VALUE_TYPE_SIGNED_LONG_LONG,
    NVML_VALUE_TYPE_UNSIGNED_INT,
    NVML_VALUE_TYPE_UNSIGNED_LONG,
    NVML_VALUE_TYPE_UNSIGNED_LONG_LONG,
    NVML_VALUE_TYPE_UNSIGNED_SHORT,
    c_nvmlFieldValue_t,
)

FILE_MAGIC = b"NVF1"
HEADER_STRUCT = struct.Struct("<4sHHi")
HOST_TS_STRUCT = struct.Struct("<Q")
FIELD_SIZE = sizeof(c_nvmlFieldValue_t)


def decode_value(field: c_nvmlFieldValue_t):
    """Return a python value extracted from the union based on valueType."""
    vt = field.valueType
    if vt == NVML_VALUE_TYPE_DOUBLE:
        return field.value.dVal
    if vt == NVML_VALUE_TYPE_UNSIGNED_INT:
        return field.value.uiVal
    if vt == NVML_VALUE_TYPE_UNSIGNED_LONG:
        return field.value.ulVal
    if vt == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
        return field.value.ullVal
    if vt == NVML_VALUE_TYPE_SIGNED_LONG_LONG:
        return field.value.sllVal
    if vt == NVML_VALUE_TYPE_SIGNED_INT:
        return field.value.siVal
    if vt == NVML_VALUE_TYPE_UNSIGNED_SHORT:
        return field.value.usVal
    return f"<unknown type {vt}>"


def read_records(path: Path, limit: int | None):
    """Generator yielding (host_ts, c_nvmlFieldValue_t, version) tuples."""
    with path.open("rb") as fh:
        header = fh.read(HEADER_STRUCT.size)
        if len(header) != HEADER_STRUCT.size:
            raise RuntimeError("File too short to contain header")
        magic, version, field_size, host_ts_size = HEADER_STRUCT.unpack(header)
        if magic != FILE_MAGIC:
            raise RuntimeError("Unrecognized file magic")
        if field_size != FIELD_SIZE:
            raise RuntimeError(f"Field size mismatch (file={field_size}, expected={FIELD_SIZE})")
        if host_ts_size != HOST_TS_STRUCT.size:
            raise RuntimeError("Host timestamp size mismatch")
        stride = host_ts_size + field_size

        count = 0
        while True:
            if limit is not None and count >= limit:
                break
            chunk = fh.read(stride)
            if len(chunk) != stride:
                break
            host_ts, = HOST_TS_STRUCT.unpack_from(chunk, 0)
            field_bytes = chunk[host_ts_size:]
            field = c_nvmlFieldValue_t.from_buffer_copy(field_bytes)
            yield host_ts, field, version
            count += 1


def get_pandas():
    global pd
    if pd is not None:
        return pd
    try:
        import pandas as _pd  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to build a CSV (pip install pandas).") from exc
    pd = _pd
    return pd


def records_to_dataframe(records):
    pandas = get_pandas()
    rows = []
    for host_ts, field, _version in records:
        rows.append(
            {
                "host_timestamp_ns": host_ts,
                "nvml_timestamp_ns": field.timestamp,
                "latency_us": field.latencyUsec,
                "field_id": field.fieldId,
                "scope_id": field.scopeId,
                "value_type": int(field.valueType),
                "nvml_return": int(field.nvmlReturn),
                "value": decode_value(field),
            }
        )
    return pandas.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Dump the first N NVML field records.")
    parser.add_argument("path", type=Path, help="Binary file produced by utilization.py")
    parser.add_argument("-n", "--count", type=int, default=10, help="Number of records to print")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum number of records to load from the file (default: all).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write all loaded records as a CSV via pandas.",
    )
    args = parser.parse_args()

    records = list(read_records(args.path, args.max_records))
    if not records:
        print("No records found.")
        return

    print_limit = min(args.count, len(records))
    for idx in range(print_limit):
        host_ts, field, _version = records[idx]
        value = decode_value(field)
        print(
            f"[#{idx:06d}] host_ts={host_ts}ns "
            f"fieldId={field.fieldId} scopeId={field.scopeId} "
            f"nvml_ts={field.timestamp}ns latency={field.latencyUsec}us "
            f"valueType={field.valueType} nvmlReturn={field.nvmlReturn} value={value}"
        )

    if print_limit < args.count:
        print(f"Only {print_limit} records available (requested {args.count}).")

    if args.csv:
        df = records_to_dataframe(records)
        df.to_csv(args.csv, index=False)
        print(f"Wrote {len(df)} rows to {args.csv}")


if __name__ == "__main__":
    main()