#!/usr/bin/env python3
"""Minimal upload API and static server for the CCL-Bench website."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

REPO_ROOT = Path(__file__).parent.parent

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _strip_none(obj):
    if isinstance(obj, dict):
        for k in list(obj):
            if obj[k] is None:
                del obj[k]
            else:
                _strip_none(obj[k])
    elif isinstance(obj, list):
        for item in obj:
            _strip_none(item)


def _run_processing(upload_id: str, job_dir: Path, results_path: Path) -> None:
    with _jobs_lock:
        _jobs[upload_id]["status"] = "processing"
    try:
        r = subprocess.run(
            [sys.executable, "website/process_upload.py",
             "--trace-dir", str(job_dir), "--output", str(results_path)],
            capture_output=True, text=True, timeout=600, cwd=str(REPO_ROOT),
        )
        for entry in list(job_dir.iterdir()):
            if entry.name == "results.json":
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
        with _jobs_lock:
            if r.returncode != 0:
                _jobs[upload_id].update({"status": "error", "error": (r.stderr.strip() or "non-zero exit")[-500:]})
            else:
                _jobs[upload_id]["status"] = "done"
                _jobs[upload_id]["log"] = r.stderr.strip()[-1000:]
    except subprocess.TimeoutExpired:
        with _jobs_lock:
            _jobs[upload_id].update({"status": "error", "error": "Processing timed out"})
    except Exception as e:
        with _jobs_lock:
            _jobs[upload_id].update({"status": "error", "error": str(e)})


def _parse_allowed_origins(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def create_app(
    static_root: Path,
    upload_root: Path,
    allowed_origins: set[str],
    max_upload_mb: int,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024

    static_root = static_root.resolve()
    upload_root = upload_root.resolve()
    upload_root.mkdir(parents=True, exist_ok=True)

    @app.after_request
    def _cors(resp):  # type: ignore[override]
        origin = request.headers.get("Origin")
        if origin and ("*" in allowed_origins or origin in allowed_origins):
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"ok": True})

    @app.route("/api/upload", methods=["OPTIONS"])
    def upload_options():
        return ("", 204)

    @app.route("/api/upload", methods=["POST"])
    def upload_file():
        files = request.files.getlist("files")
        group_name = (request.form.get("group_name") or "").strip()
        workload_card_json = request.form.get("workload_card") or ""

        if not files or all(not f.filename for f in files):
            return jsonify({"error": "No files provided"}), 400
        if not group_name:
            return jsonify({"error": "Missing group_name"}), 400

        upload_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
        job_dir = upload_root / upload_id
        job_dir.mkdir(parents=True, exist_ok=True)

        uploaded, rejected = [], []
        for f in files:
            safe = secure_filename(f.filename or "")
            if not safe:
                rejected.append({"original": f.filename, "reason": "invalid filename"})
                continue
            dest = job_dir / safe
            f.save(dest)
            uploaded.append({"filename": safe, "size_bytes": dest.stat().st_size})

        if not uploaded:
            return jsonify({"error": "All files were rejected", "rejected": rejected}), 400

        yaml_written = False
        if workload_card_json and _HAS_YAML:
            try:
                card = json.loads(workload_card_json)
                _strip_none(card)
                (job_dir / (secure_filename(group_name) + ".yaml")).write_text(
                    _yaml.dump(card, default_flow_style=False, allow_unicode=True)
                )
                yaml_written = True
            except Exception:
                pass

        with _jobs_lock:
            _jobs[upload_id] = {"status": "queued"}
        threading.Thread(
            target=_run_processing,
            args=(upload_id, job_dir, job_dir / "results.json"),
            daemon=True,
        ).start()

        return jsonify({
            "status": "ok",
            "upload_id": upload_id,
            "group_name": group_name,
            "uploaded": uploaded,
            "rejected": rejected,
            "workload_card_generated": yaml_written,
        })

    @app.route("/api/results/<upload_id>", methods=["GET", "OPTIONS"])
    def get_results(upload_id: str):
        if request.method == "OPTIONS":
            return ("", 204)
        with _jobs_lock:
            job = _jobs.get(upload_id)
        results_path = upload_root / upload_id / "results.json"
        if job is None:
            if results_path.exists():
                return jsonify({"status": "done", "results": json.loads(results_path.read_text())})
            return jsonify({"error": "Upload ID not found"}), 404
        status = job["status"]
        if status == "done" and results_path.exists():
            return jsonify({"status": "done", "results": json.loads(results_path.read_text()), "log": job.get("log", "")})
        if status == "error":
            return jsonify({"status": "error", "error": job.get("error", "Unknown error")})
        return jsonify({"status": status})

    @app.route("/uploads/<path:filename>", methods=["GET"])
    def get_uploaded_file(filename: str):
        return send_from_directory(str(upload_root), filename, as_attachment=False)

    @app.route("/", methods=["GET"])
    def serve_root():
        return send_from_directory(str(static_root), "index.html")

    @app.route("/<path:path>", methods=["GET"])
    def serve_static(path: str):
        candidate = static_root / path
        if not candidate.is_file():
            abort(404)
        return send_from_directory(str(static_root), path)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCL-Bench upload/static server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--static-root",
        default=".",
        help="Directory containing index.html and website/ assets.",
    )
    parser.add_argument(
        "--upload-dir",
        default="uploaded_files",
        help="Directory to store uploaded files.",
    )
    parser.add_argument(
        "--max-upload-mb",
        type=int,
        default=200,
        help="Maximum upload size in MB.",
    )
    parser.add_argument(
        "--allowed-origins",
        default=os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"),
        help="Comma-separated origins allowed to call /api/upload. Use '*' to allow all.",
    )
    args = parser.parse_args()

    app = create_app(
        static_root=Path(args.static_root),
        upload_root=Path(args.upload_dir),
        allowed_origins=_parse_allowed_origins(args.allowed_origins),
        max_upload_mb=args.max_upload_mb,
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
