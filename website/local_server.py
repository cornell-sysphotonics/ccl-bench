#!/usr/bin/env python3
"""
Serve the CCL-Bench website with a local SQLite query API.

Run from the repository root:
    python website/local_server.py

Then open:
    http://127.0.0.1:8081/

The static site still works without this server through website/data.js. When
served by this script, index.html detects /api/bootstrap and uses SQLite-backed
filtering/sorting instead.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sqlite3
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "website" / "benchmark_data.sqlite"

METADATA_JSON_FIELDS = {
    "comm_env": "comm_env_json",
    "protocol_selection": "protocol_selection_json",
    "trace_types": "trace_types_json",
    "metric_traces": "metric_traces_json",
}

METADATA_FIELDS = [
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

SORT_COLUMNS = {
    "workload_name": "m.workload_name",
    "model_family": "m.model_family",
    "phase": "m.phase",
    "precision": "m.precision",
    "hardware_type": "m.hardware_type",
    "hardware_model": "m.hardware_model",
    "__accel": "m.total_count",
    "framework": "m.framework",
    "compiler": "m.compiler",
    "tp": "m.tp",
    "pp": "m.pp",
    "dp_replicate": "m.dp_replicate",
    "dp_shard": "m.dp_shard",
    "ep": "m.ep",
    "comm_library": "m.comm_library",
    "batch_size": "m.batch_size",
    "seq_len": "m.seq_len",
}


def parse_json(value: str | None, fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def metric_value(num, text):
    return num if num is not None else text


class BenchmarkStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def metric_info(self) -> dict:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT metric_name, label, category, unit, higher_is_better,
                       description, trace_types_json, phases_json
                FROM metric_info
                ORDER BY metric_name
                """
            ).fetchall()
        return {
            row["metric_name"]: {
                "label": row["label"] or row["metric_name"],
                "category": row["category"],
                "unit": row["unit"],
                "higher_is_better": bool(row["higher_is_better"]),
                "description": row["description"],
                "trace_types": parse_json(row["trace_types_json"], []),
                "phases": parse_json(row["phases_json"], []),
            }
            for row in rows
        }

    def metric_categories(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT category_id, label
                FROM metric_categories
                ORDER BY sort_order, category_id
                """
            ).fetchall()
        return [{"id": row["category_id"], "label": row["label"]} for row in rows]

    def all_metrics(self) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT metric_name FROM metric_info ORDER BY metric_name"
            ).fetchall()
        return [row["metric_name"] for row in rows]

    def generated_at(self) -> str:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT value FROM run_info WHERE key = 'generated_at'"
            ).fetchone()
        return row["value"] if row else ""

    def query_rows(self, params: dict) -> dict:
        search = (params.get("search") or "").strip().lower()
        sort_key = params.get("sort_key") or "workload_name"
        sort_dir = "DESC" if str(params.get("sort_dir")) == "-1" else "ASC"
        limit = min(max(int(params.get("limit") or 1000), 1), 5000)

        metric_sort = sort_key.startswith("__m__")
        metric_name = sort_key[5:] if metric_sort else None
        sort_expr = "x.metric_value_num" if metric_sort else SORT_COLUMNS.get(sort_key, "m.workload_name")

        where = []
        bind = {}
        if search:
            bind["search"] = f"%{search}%"
            where.append(
                """
                (
                    lower(coalesce(m.workload_name, '')) LIKE :search OR
                    lower(coalesce(m.model_family, '')) LIKE :search OR
                    lower(coalesce(m.phase, '')) LIKE :search OR
                    lower(coalesce(m.hardware_type, '')) LIKE :search OR
                    lower(coalesce(m.hardware_model, '')) LIKE :search OR
                    lower(coalesce(m.framework, '')) LIKE :search OR
                    lower(coalesce(m.comm_library, '')) LIKE :search OR
                    EXISTS (
                        SELECT 1 FROM metrics sm
                        WHERE sm.row_id = m.row_id
                          AND lower(coalesce(sm.metric_value_text, CAST(sm.metric_value_num AS TEXT), '')) LIKE :search
                    )
                )
                """
            )

        join = ""
        if metric_sort:
            bind["metric_name"] = metric_name
            join = "LEFT JOIN metrics x ON x.row_id = m.row_id AND x.metric_name = :metric_name"

        where_sql = "WHERE " + " AND ".join(where) if where else ""
        sql = f"""
            SELECT m.*
            FROM metadata m
            {join}
            {where_sql}
            ORDER BY {sort_expr} IS NULL ASC, {sort_expr} {sort_dir}, m.workload_name COLLATE NOCASE ASC
            LIMIT :limit
        """
        bind["limit"] = limit

        with self.connect() as conn:
            meta_rows = conn.execute(sql, bind).fetchall()
            total = conn.execute(
                f"SELECT COUNT(*) FROM metadata m {where_sql}",
                {k: v for k, v in bind.items() if k != "limit" and k != "metric_name"},
            ).fetchone()[0]
            row_ids = [row["row_id"] for row in meta_rows]
            metrics_by_row = {row_id: {} for row_id in row_ids}
            if row_ids:
                placeholders = ",".join("?" for _ in row_ids)
                metric_rows = conn.execute(
                    f"""
                    SELECT row_id, metric_name, metric_value_num, metric_value_text
                    FROM metrics
                    WHERE row_id IN ({placeholders})
                    """,
                    row_ids,
                ).fetchall()
                for row in metric_rows:
                    metrics_by_row[row["row_id"]][row["metric_name"]] = metric_value(
                        row["metric_value_num"],
                        row["metric_value_text"],
                    )

        rows = []
        for meta_row in meta_rows:
            metadata = {field: meta_row[field] for field in METADATA_FIELDS}
            metadata["moe"] = bool(metadata["moe"])
            for public_name, db_name in METADATA_JSON_FIELDS.items():
                fallback = {} if public_name == "comm_env" else []
                metadata[public_name] = parse_json(meta_row[db_name], fallback)

            rows.append(
                {
                    "trace": meta_row["trace"],
                    "metadata": metadata,
                    "metrics": metrics_by_row.get(meta_row["row_id"], {}),
                    "__idx": meta_row["row_id"] - 1,
                }
            )

        return {"rows": rows, "total": total}

    def bootstrap(self) -> dict:
        payload = {
            "generated_at": self.generated_at(),
            "all_metrics": self.all_metrics(),
            "metric_categories": self.metric_categories(),
            "metric_info": self.metric_info(),
        }
        payload.update(self.query_rows({"limit": "5000"}))
        return payload


class Handler(SimpleHTTPRequestHandler):
    store: BenchmarkStore

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/bootstrap":
            self.send_json(self.store.bootstrap())
            return
        if parsed.path == "/api/rows":
            query = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
            self.send_json(self.store.query_rows(query))
            return
        if parsed.path == "/api/health":
            self.send_json({"ok": True, "db": str(self.store.db_path)})
            return
        super().do_GET()

    def send_json(self, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def guess_type(self, path: str) -> str:
        if path.endswith(".sqlite"):
            return "application/octet-stream"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve CCL-Bench with a SQLite API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    db_path = args.db.expanduser().resolve()
    if not db_path.exists():
        raise SystemExit(
            f"Missing {db_path}. Run `python website/generate_database.py` first."
        )

    Handler.store = BenchmarkStore(db_path)
    os.chdir(ROOT)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving CCL-Bench at http://{args.host}:{args.port}/")
    print(f"SQLite database: {db_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
