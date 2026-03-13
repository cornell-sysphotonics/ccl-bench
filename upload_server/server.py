#!/usr/bin/env python3
"""
CCL-Bench Upload Server (Hardened)
==================================

Production-leaning Flask server for receiving public uploads safely.

API
---
POST /api/upload
    multipart/form-data
    Fields:
        files       — one or more files (required)
        group_name  — subfolder name (optional)
        description — free-text note (optional)
        workload_card — JSON string (optional)

GET /api/uploads
    Lists uploaded groups and manifests.

GET /api/health
    Health-check.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import safe_join, secure_filename

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# =============================================================================
# Configuration
# =============================================================================

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

def _env_int(name: str, default: int, min_value: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if min_value is not None and value < min_value:
        return default
    return value

# Base paths
THIS_FILE = Path(__file__).resolve()
APP_DIR = THIS_FILE.parent
REPO_ROOT = APP_DIR.parent

# IMPORTANT: Prefer placing uploads OUTSIDE web root in production
DEFAULT_UPLOAD_DIR = APP_DIR / "uploads"

UPLOAD_FOLDER = Path(os.getenv("CCL_UPLOAD_DIR", str(DEFAULT_UPLOAD_DIR))).resolve()

# Request limits
MAX_CONTENT_LENGTH = _env_int("CCL_MAX_CONTENT_LENGTH_BYTES", 500 * 1024 * 1024, 1)  # 500MB
MAX_FILES_PER_REQUEST = _env_int("CCL_MAX_FILES_PER_REQUEST", 20, 1)
MAX_FILENAME_LENGTH = _env_int("CCL_MAX_FILENAME_LENGTH", 255, 8)
MAX_GROUP_NAME_LENGTH = _env_int("CCL_MAX_GROUP_NAME_LENGTH", 120, 8)
MAX_DESCRIPTION_LENGTH = _env_int("CCL_MAX_DESCRIPTION_LENGTH", 10_000, 0)
MAX_WORKLOAD_CARD_BYTES = _env_int("CCL_MAX_WORKLOAD_CARD_BYTES", 1_000_000, 0)  # 1MB JSON payload

# Allowed extensions (lowercase, normalized)
ALLOWED_EXTENSIONS = {
    # Trace formats
    ".nsys-rep", ".json",
    # Archives
    ".tar", ".gz", ".tgz", ".zip", ".bz2", ".xz", ".zst",
}

# Optional stricter CORS; comma-separated list of origins. Example:
# CCL_ALLOWED_ORIGINS="https://cclbench.ai,https://www.cclbench.ai"
# If unset, defaults to no cross-origin access unless explicitly allowed.
ALLOWED_ORIGINS_ENV = os.getenv("CCL_ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS_ENV.split(",") if o.strip()]

# Runtime options
TRUST_PROXY = _env_bool("CCL_TRUST_PROXY", True)
LOG_LEVEL = os.getenv("CCL_LOG_LEVEL", "INFO").upper()

# Flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# If behind nginx / reverse proxy / load balancer
if TRUST_PROXY:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)  # type: ignore[assignment]

# CORS configuration
if ALLOWED_ORIGINS:
    CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})
else:
    # Allow all origins by default so the public website (e.g. cclbench.ai on
    # GitHub Pages) can reach the upload API running on a different host/port.
    # Tighten this with CCL_ALLOWED_ORIGINS in production.
    CORS(app, resources={r"/api/*": {"origins": "*"}})

# Logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ccl_upload_server")


# =============================================================================
# Helpers
# =============================================================================

_GROUP_NAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")

def json_error(message: str, status_code: int = 400, **extra: Any):
    payload = {"status": "error", "message": message}
    payload.update(extra)
    return jsonify(payload), status_code

def normalize_extension(filename: str) -> str:
    # Handle compound names with pathlib; only last suffix used by allowlist here
    return Path(filename).suffix.lower()

def allowed_file(filename: str) -> bool:
    return normalize_extension(filename) in ALLOWED_EXTENSIONS

def sanitize_group_name(name: str) -> str:
    """
    Produce a stable, short, filesystem-safe name (not empty).
    Uses secure_filename plus regex normalization to avoid weird edge cases.
    """
    name = (name or "").strip()
    if not name:
        return ""
    name = name[:MAX_GROUP_NAME_LENGTH]
    name = secure_filename(name)
    name = _GROUP_NAME_SAFE_RE.sub("-", name).strip("._-")
    return name[:MAX_GROUP_NAME_LENGTH]

def strip_nulls(obj: Any) -> Any:
    """Recursively remove keys with None / empty-string / empty-list values."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            sv = strip_nulls(v)
            if sv is not None and sv != "" and sv != []:
                out[k] = sv
        return out
    if isinstance(obj, list):
        return [strip_nulls(i) for i in obj if i is not None and i != ""]
    return obj

def atomic_write_bytes(path: Path, data: bytes) -> None:
    """
    Atomic file write to avoid partial/corrupt files on crashes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)

def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))

def safe_mkdir_no_symlink(path: Path) -> None:
    """
    Create directory and reject if final path resolves to a symlink.
    """
    path.mkdir(parents=True, exist_ok=True)
    try:
        if path.is_symlink():
            raise RuntimeError("Upload directory path is a symlink; refusing to write.")
    except OSError:
        raise RuntimeError("Could not validate upload directory.")

def safe_file_dest(upload_dir: Path, filename: str) -> Path:
    """
    Construct destination path safely and ensure it stays under upload_dir.
    """
    # We already secure_filename() before calling this, but double-check.
    candidate = upload_dir / filename
    resolved_parent = upload_dir.resolve()
    # candidate may not exist yet; resolve strict=False
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_parent)
    except ValueError:
        raise RuntimeError("Resolved path escapes upload directory.")
    return candidate

def unique_dest_path(upload_dir: Path, filename: str) -> Path:
    """
    Avoid overwrite by appending a short UUID suffix.
    """
    base_dest = safe_file_dest(upload_dir, filename)
    if not base_dest.exists():
        return base_dest

    p = Path(filename)
    stem = p.stem
    suffix = p.suffix
    for _ in range(20):
        candidate_name = f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
        candidate = safe_file_dest(upload_dir, candidate_name)
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not generate unique filename after multiple attempts.")

def save_upload_manifest(
    upload_dir: Path,
    upload_id: str,
    file_records: List[Dict[str, Any]],
    group_name: str,
    description: str,
) -> None:
    manifest = {
        "upload_id": upload_id,
        "group_name": group_name,
        "description": description,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "files": file_records,
    }
    manifest_path = upload_dir / "upload_manifest.json"

    # Append to existing manifest list robustly
    manifests: List[Dict[str, Any]] = []
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                manifests = [m for m in existing if isinstance(m, dict)]
            elif isinstance(existing, dict):
                manifests = [existing]
        except Exception:
            logger.warning("Failed to parse existing manifest; recreating %s", manifest_path)

    manifests.append(manifest)
    atomic_write_text(manifest_path, json.dumps(manifests, indent=2, ensure_ascii=False) + "\n")

def write_workload_card(upload_dir: Path, group_name: str, card_data: Dict[str, Any]) -> str:
    """
    Write workload card as YAML if PyYAML is available, else JSON fallback.
    """
    card_data = strip_nulls(card_data)
    if HAS_YAML:
        card_path = upload_dir / f"{group_name}.yaml"
        # safe_dump avoids arbitrary Python object tags
        yaml_text = yaml.safe_dump(
            card_data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )
        atomic_write_text(card_path, yaml_text)
    else:
        card_path = upload_dir / f"{group_name}.json"
        atomic_write_text(card_path, json.dumps(card_data, indent=2, ensure_ascii=False) + "\n")
    return card_path.name

def validate_and_parse_workload_card(raw: str) -> Dict[str, Any]:
    if len(raw.encode("utf-8", errors="ignore")) > MAX_WORKLOAD_CARD_BYTES:
        raise ValueError(f"workload_card too large (>{MAX_WORKLOAD_CARD_BYTES} bytes)")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid workload_card JSON: {e.msg}") from e
    if not isinstance(parsed, dict):
        raise ValueError("workload_card must be a JSON object")
    return parsed

def secure_uploaded_filename(original_name: str) -> Tuple[str | None, str | None]:
    """
    Returns (safe_name, reject_reason). Ensures name is non-empty, length-limited,
    and extension is allowed.
    """
    if not original_name:
        return None, "Empty filename."

    # Drop path components from malicious clients before secure_filename
    original_basename = Path(original_name).name

    fname = secure_filename(original_basename)
    if not fname:
        return None, "Invalid filename."

    if len(fname) > MAX_FILENAME_LENGTH:
        # preserve suffix if possible
        p = Path(fname)
        suffix = p.suffix
        cutoff = MAX_FILENAME_LENGTH - len(suffix)
        if cutoff <= 0:
            return None, "Filename too long."
        fname = p.stem[:cutoff] + suffix

    # Hidden-ish names after sanitization
    if fname in {".", ".."} or fname.startswith("."):
        return None, "Hidden or invalid filename."

    if not allowed_file(fname):
        return None, (
            "Extension not allowed. Allowed: " + ", ".join(sorted(ALLOWED_EXTENSIONS))
        )

    return fname, None

def stream_save_filestorage(fs, dest: Path) -> int:
    """
    Save uploaded file to disk using a temporary file then atomic rename.
    Returns final size in bytes.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(dest.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            fs.save(tmp)  # writes stream to open file object
            tmp.flush()
            os.fsync(tmp.fileno())
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise
    os.replace(tmp_path, dest)
    return dest.stat().st_size

def ensure_uploads_dir() -> None:
    safe_mkdir_no_symlink(UPLOAD_FOLDER)


# =============================================================================
# Security headers / hooks
# =============================================================================

@app.after_request
def add_security_headers(response):
    # Reasonable defaults for a simple API/static server
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    # If TLS terminated upstream, HSTS is best set at the reverse proxy.
    return response


# =============================================================================
# Error handlers
# =============================================================================

@app.errorhandler(RequestEntityTooLarge)
def handle_413(_e):
    return json_error(
        f"Request too large. Max allowed is {MAX_CONTENT_LENGTH} bytes.",
        status_code=413,
    )

@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException):
    # Keep JSON for API routes; HTML/plain for static routes if desired.
    if request.path.startswith("/api/"):
        return json_error(e.description or "HTTP error", status_code=e.code or 500)
    return e

@app.errorhandler(Exception)
def handle_unexpected_exception(e: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.path)
    if request.path.startswith("/api/"):
        return json_error("Internal server error", status_code=500)
    return "Internal server error", 500


# =============================================================================
# Routes
# =============================================================================

@app.route("/")
def index():
    """
    Serve main CCL-Bench index.html from repo root.
    """
    index_path = REPO_ROOT / "index.html"
    if not index_path.is_file():
        abort(404, description="index.html not found")
    return send_file(str(index_path))

@app.route("/favicon.ico")
def favicon():
    favicon_path = REPO_ROOT / "favicon.ico"
    if favicon_path.exists():
        return send_from_directory(str(REPO_ROOT), "favicon.ico", mimetype="image/x-icon")
    return "", 204

@app.route("/<path:filepath>")
def serve_static(filepath: str):
    """
    Serve static assets from repo root safely.
    Prevents path traversal and blocks sensitive/internal paths.
    """
    # Quick denylist for obvious internals
    blocked_prefixes = (
        "upload_server/",
        "uploads/",
        ".git/",
        ".env",
        ".venv/",
        "__pycache__/",
    )
    if filepath.startswith(blocked_prefixes) or "/." in filepath or filepath.startswith("."):
        abort(404)

    # safe_join prevents traversal such as ../../etc/passwd
    safe_path = safe_join(str(REPO_ROOT), filepath)
    if safe_path is None:
        abort(404)

    full_path = Path(safe_path)
    if full_path.is_file():
        # send_from_directory handles conditional responses efficiently
        return send_from_directory(str(REPO_ROOT), filepath)

    abort(404)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "time_utc": datetime.now(timezone.utc).isoformat(),
    })

@app.route("/api/upload", methods=["POST"])
def upload_files():
    """
    Accept one or more files and save them under uploads/<group_name>__<timestamp>/.
    """
    ensure_uploads_dir()

    # Basic content-type sanity check (multipart/form-data expected)
    ctype = request.content_type or ""
    if "multipart/form-data" not in ctype.lower():
        return json_error("Content-Type must be multipart/form-data", 400)

    # Validate files field presence
    if "files" not in request.files:
        return json_error("No files provided. Use the 'files' form field.", 400)

    files = request.files.getlist("files")
    if not files:
        return json_error("No files provided.", 400)
    if len(files) > MAX_FILES_PER_REQUEST:
        return json_error(
            f"Too many files. Max per request is {MAX_FILES_PER_REQUEST}.",
            400,
        )

    if all((not f or not (f.filename or "").strip()) for f in files):
        return json_error("No selected files.", 400)

    # Metadata
    group_name_raw = (request.form.get("group_name", "") or "").strip()
    description = (request.form.get("description", "") or "").strip()

    if len(description) > MAX_DESCRIPTION_LENGTH:
        return json_error(
            f"description too long (max {MAX_DESCRIPTION_LENGTH} chars).",
            400,
        )

    upload_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    safe_group_base = sanitize_group_name(group_name_raw)
    if safe_group_base:
        safe_group = f"{safe_group_base}__{timestamp}"
    else:
        safe_group = f"upload-{upload_id[:8]}__{timestamp}"

    # Final path safety
    upload_dir = (UPLOAD_FOLDER / safe_group).resolve(strict=False)
    try:
        upload_dir.relative_to(UPLOAD_FOLDER.resolve())
    except ValueError:
        return json_error("Resolved upload path is invalid.", 500)

    # Create dir (reject if symlink)
    try:
        safe_mkdir_no_symlink(upload_dir)
    except Exception as e:
        logger.exception("Failed creating upload dir")
        return json_error(f"Failed to initialize upload directory: {e}", 500)

    saved: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []

    # Save files
    for fs in files:
        if not fs or not (fs.filename or "").strip():
            continue

        safe_name, reject_reason = secure_uploaded_filename(fs.filename)
        if reject_reason:
            rejected.append({"original": fs.filename or "", "reason": reject_reason})
            continue

        assert safe_name is not None  # for type checkers
        try:
            dest = unique_dest_path(upload_dir, safe_name)
            file_size = stream_save_filestorage(fs, dest)

            # Optional post-save sanity checks
            if file_size <= 0:
                # Remove empty file uploads if you consider them invalid
                dest.unlink(missing_ok=True)
                rejected.append({"original": fs.filename or "", "reason": "Empty file."})
                continue

            saved.append({
                "filename": dest.name,
                "size_bytes": file_size,
            })
        except Exception as e:
            logger.warning("Failed to save file %r: %s", fs.filename, e)
            rejected.append({"original": fs.filename or "", "reason": "Failed to save file."})

    if not saved:
        # Clean up empty directory if nothing was saved
        try:
            shutil.rmtree(upload_dir, ignore_errors=True)
        except Exception:
            pass
        return json_error("No files were saved.", 400, rejected=rejected)

    # Optional workload card
    workload_card_generated = False
    wc_raw = (request.form.get("workload_card", "") or "").strip()
    if wc_raw:
        try:
            card_data = validate_and_parse_workload_card(wc_raw)
            card_filename = write_workload_card(upload_dir, safe_group, card_data)
            workload_card_generated = True
            card_path = upload_dir / card_filename
            saved.append({
                "filename": card_filename,
                "size_bytes": card_path.stat().st_size if card_path.exists() else None,
            })
        except Exception as e:
            logger.warning("Failed to generate workload card: %s", e)
            rejected.append({"original": "workload_card", "reason": str(e)})

    # Manifest write (best-effort but should usually succeed)
    try:
        save_upload_manifest(upload_dir, upload_id, saved, safe_group, description)
    except Exception:
        logger.exception("Failed writing manifest for upload_id=%s", upload_id)
        # Upload itself succeeded; return success with warning
        return jsonify({
            "status": "ok",
            "upload_id": upload_id,
            "group_name": safe_group,
            "uploaded": saved,
            "rejected": rejected,
            "workload_card_generated": workload_card_generated,
            "warning": "Files uploaded, but manifest write failed.",
        }), 200

    return jsonify({
        "status": "ok",
        "upload_id": upload_id,
        "group_name": safe_group,
        "uploaded": saved,
        "rejected": rejected,
        "workload_card_generated": workload_card_generated,
    }), 200

@app.route("/api/uploads", methods=["GET"])
def list_uploads():
    """
    List uploaded groups and manifest metadata.
    NOTE: If public, this endpoint can expose user-supplied descriptions/filenames.
    Consider auth/rate limiting in production.
    """
    ensure_uploads_dir()

    groups = []
    # Sort newest directories first if names include timestamp suffix
    for entry in sorted(UPLOAD_FOLDER.iterdir(), reverse=True):
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue

        # Skip symlinked dirs for safety
        try:
            if entry.is_symlink():
                continue
        except OSError:
            continue

        manifest_path = entry / "upload_manifest.json"
        manifest = None
        if manifest_path.exists() and manifest_path.is_file():
            try:
                # Avoid loading arbitrarily huge manifest files
                if manifest_path.stat().st_size <= 5 * 1024 * 1024:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                else:
                    manifest = {"warning": "manifest too large to display"}
            except Exception:
                manifest = {"warning": "manifest unreadable"}

        try:
            file_count = sum(
                1 for f in entry.iterdir()
                if f.is_file() and f.name != "upload_manifest.json"
            )
        except Exception:
            file_count = None

        groups.append({
            "group_name": entry.name,
            "file_count": file_count,
            "manifest": manifest,
        })

    return jsonify({"status": "ok", "groups": groups})


# =============================================================================
# Main (development only)
# =============================================================================

if __name__ == "__main__":
    ensure_uploads_dir()

    # NEVER use debug=True in public deployment.
    host = os.getenv("HOST", "0.0.0.0")
    port = _env_int("PORT", 5000, 1)
    debug = _env_bool("FLASK_DEBUG", False)

    logger.info("Starting CCL upload server on %s:%d (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug)