#!/usr/bin/env python3
"""
Generate a normalized SQLite database for the CCL-Bench website.

Run from the repository root after website/generate_data.py:
    python website/generate_database.py

Input:
    website/benchmark_data.json

Output:
    website/benchmark_data.sqlite

This is intentionally compatible with the static path. index.html uses the
SQLite API when served by website/local_server.py and falls back to data.js
when opened statically.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "website" / "benchmark_data.json"
DB_PATH = ROOT / "website" / "benchmark_data.sqlite"


METADATA_COLUMNS = [
    "workload_name",
    "description",
    "hf_url",
    "trace_url",
    "model_family",
    "phase",
    "precision",
    "moe",
    "granularity",
    "epochs",
    "iteration",
    "batch_size",
    "seq_len",
    "input_len",
    "output_len",
    "dataset",
    "hardware_type",
    "hardware_model",
    "total_count",
    "count_per_node",
    "network_topology",
    "bandwidth_scaleout",
    "bandwidth_scaleup",
    "driver_version",
    "framework",
    "compiler",
    "tp",
    "pp",
    "pp_mb",
    "dp_replicate",
    "dp_shard",
    "ep",
    "cp",
    "comm_library",
    "comm_library_ver",
]


def scalar_value(value: Any) -> Any:
    """Convert JSON values into SQLite-friendly scalar values."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (str, int, float)) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


def numeric_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;

        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS metrics;
        DROP TABLE IF EXISTS metric_info;
        DROP TABLE IF EXISTS metric_categories;
        DROP TABLE IF EXISTS run_info;

        CREATE TABLE run_info (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE metadata (
            row_id INTEGER PRIMARY KEY,
            trace TEXT NOT NULL UNIQUE,
            workload_name TEXT,
            description TEXT,
            hf_url TEXT,
            trace_url TEXT,
            model_family TEXT,
            phase TEXT,
            precision TEXT,
            moe INTEGER,
            granularity TEXT,
            epochs REAL,
            iteration REAL,
            batch_size REAL,
            seq_len REAL,
            input_len REAL,
            output_len REAL,
            dataset TEXT,
            hardware_type TEXT,
            hardware_model TEXT,
            total_count REAL,
            count_per_node REAL,
            network_topology TEXT,
            bandwidth_scaleout REAL,
            bandwidth_scaleup REAL,
            driver_version TEXT,
            framework TEXT,
            compiler TEXT,
            tp REAL,
            pp REAL,
            pp_mb REAL,
            dp_replicate REAL,
            dp_shard REAL,
            ep REAL,
            cp REAL,
            comm_library TEXT,
            comm_library_ver TEXT,
            comm_env_json TEXT,
            protocol_selection_json TEXT,
            trace_types_json TEXT,
            metric_traces_json TEXT
        );

        CREATE TABLE metrics (
            row_id INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value_num REAL,
            metric_value_text TEXT,
            PRIMARY KEY (row_id, metric_name),
            FOREIGN KEY (row_id) REFERENCES metadata(row_id)
        );

        CREATE TABLE metric_info (
            metric_name TEXT PRIMARY KEY,
            label TEXT,
            category TEXT,
            unit TEXT,
            higher_is_better INTEGER,
            description TEXT,
            trace_types_json TEXT,
            phases_json TEXT
        );

        CREATE TABLE metric_categories (
            category_id TEXT PRIMARY KEY,
            label TEXT,
            sort_order INTEGER
        );

        CREATE INDEX idx_metadata_phase ON metadata(phase);
        CREATE INDEX idx_metadata_model ON metadata(model_family);
        CREATE INDEX idx_metadata_comm ON metadata(comm_library);
        CREATE INDEX idx_metrics_name_value ON metrics(metric_name, metric_value_num);
        """
    )


def insert_data(conn: sqlite3.Connection, data: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO run_info(key, value) VALUES (?, ?)",
        ("generated_at", data.get("generated_at", "")),
    )
    conn.execute(
        "INSERT INTO run_info(key, value) VALUES (?, ?)",
        ("source", "website/benchmark_data.json"),
    )

    for sort_order, category in enumerate(data.get("metric_categories", [])):
        conn.execute(
            """
            INSERT INTO metric_categories(category_id, label, sort_order)
            VALUES (?, ?, ?)
            """,
            (
                category.get("id"),
                category.get("label"),
                sort_order,
            ),
        )

    for name, info in (data.get("metric_info") or {}).items():
        conn.execute(
            """
            INSERT INTO metric_info(
                metric_name, label, category, unit, higher_is_better,
                description, trace_types_json, phases_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                info.get("label") or name,
                info.get("category"),
                info.get("unit"),
                int(bool(info.get("higher_is_better"))),
                info.get("description"),
                json.dumps(info.get("trace_types") or []),
                json.dumps(info.get("phases") or []),
            ),
        )

    metadata_sql = f"""
        INSERT INTO metadata(
            row_id, trace, {", ".join(METADATA_COLUMNS)},
            comm_env_json, protocol_selection_json, trace_types_json, metric_traces_json
        )
        VALUES ({", ".join("?" for _ in range(2 + len(METADATA_COLUMNS) + 4))})
    """

    numeric_columns = {
        "epochs",
        "iteration",
        "batch_size",
        "seq_len",
        "input_len",
        "output_len",
        "total_count",
        "count_per_node",
        "bandwidth_scaleout",
        "bandwidth_scaleup",
        "tp",
        "pp",
        "pp_mb",
        "dp_replicate",
        "dp_shard",
        "ep",
        "cp",
    }

    for row_id, row in enumerate(data.get("rows", []), start=1):
        meta = row.get("metadata") or {}
        values = []
        for col in METADATA_COLUMNS:
            value = meta.get(col)
            if col in numeric_columns:
                values.append(numeric_or_none(value))
            elif col == "moe":
                values.append(int(bool(value)))
            else:
                values.append(scalar_value(value))

        conn.execute(
            metadata_sql,
            [
                row_id,
                row.get("trace"),
                *values,
                json.dumps(meta.get("comm_env") or {}, sort_keys=True),
                json.dumps(meta.get("protocol_selection") or []),
                json.dumps(meta.get("trace_types") or []),
                json.dumps(meta.get("metric_traces") or []),
            ],
        )

        for metric_name, metric_value in (row.get("metrics") or {}).items():
            metric_num = numeric_or_none(metric_value)
            metric_text = None if metric_num is not None else scalar_value(metric_value)
            conn.execute(
                """
                INSERT INTO metrics(row_id, metric_name, metric_value_num, metric_value_text)
                VALUES (?, ?, ?, ?)
                """,
                (row_id, metric_name, metric_num, metric_text),
            )


def main() -> None:
    if not JSON_PATH.exists():
        raise SystemExit(f"Missing {JSON_PATH}. Run website/generate_data.py first.")

    data = json.loads(JSON_PATH.read_text())
    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as conn:
        create_schema(conn)
        insert_data(conn, data)
        conn.commit()
        conn.execute("VACUUM")

    with sqlite3.connect(DB_PATH) as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
        metric_count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]

    print(f"Wrote {DB_PATH}")
    print(f"Rows: {row_count}")
    print(f"Metric values: {metric_count}")


if __name__ == "__main__":
    main()
