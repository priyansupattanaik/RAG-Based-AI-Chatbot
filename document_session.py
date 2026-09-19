"""
Active uploaded-PDF session.

Chat, retrieval, and ingest-from-default-path are blocked until a user
uploads a PDF through this module (Streamlit or POST /upload).
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import os
import config

UPLOAD_DIR = Path(config.UPLOAD_DIR)
STATE_PATH = UPLOAD_DIR / "active.json"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
_LOCK = threading.Lock()
NO_DOCUMENT_DETAIL = (
    "No PDF has been uploaded. Upload a PDF first; the assistant will not answer "
    "questions until a document is indexed."
)


class DocumentNotReady(Exception):
    """Raised when chat/retrieval is attempted without an uploaded PDF."""


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    base = Path(name or "document.pdf").name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    if base in {".pdf", "pdf"}:
        base = "document.pdf"
    return base[:180]


def _path_in_uploads(path: Path) -> bool:
    try:
        path.resolve().relative_to(UPLOAD_DIR.resolve())
        return True
    except (OSError, ValueError):
        return False


def get_active_document() -> Optional[Dict[str, Any]]:
    if not STATE_PATH.exists():
        return None
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    path = Path(state.get("path", ""))
    if not path.is_file() or not _path_in_uploads(path):
        return None
    return state


def is_document_ready() -> bool:
    state = get_active_document()
    return bool(state and state.get("ingested") and Path(state["path"]).is_file())


def require_document_ready() -> Dict[str, Any]:
    state = get_active_document()
    if not state or not state.get("ingested") or not Path(state["path"]).is_file():
        raise DocumentNotReady(NO_DOCUMENT_DETAIL)
    return state


def clear_active_document() -> None:
    if STATE_PATH.exists():
        try:
            STATE_PATH.unlink()
        except OSError:
            pass


def save_uploaded_pdf(filename: str, data: bytes) -> Dict[str, Any]:
    if not data:
        raise ValueError("Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"PDF exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    original = (filename or "").strip()
    if not original.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are accepted.")
    if not data.startswith(b"%PDF"):
        raise ValueError("File is not a valid PDF.")

    _ensure_dirs()
    safe = f"{uuid.uuid4().hex[:8]}_{_safe_filename(original)}"
    dest = UPLOAD_DIR / safe
    if not _path_in_uploads(dest):
        raise ValueError("Refusing to store PDF outside the upload directory.")
    dest.write_bytes(data)
    state = {
        "filename": original,
        "stored_name": safe,
        "path": str(dest.resolve()),
        "size_bytes": len(data),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "ingested": False,
        "total_pages": 0,
        "total_chunks": 0,
    }
    _write_state(state)
    return state


def mark_ingested(
    total_pages: int,
    total_chunks: int,
    target_store: str = "",
    vector_db_type: str = "local",
) -> Dict[str, Any]:
    state = get_active_document()
    if not state:
        raise DocumentNotReady(NO_DOCUMENT_DETAIL)
    state["ingested"] = True
    state["total_pages"] = int(total_pages)
    state["total_chunks"] = int(total_chunks)
    state["target_store"] = target_store
    state["vector_db_type"] = vector_db_type or "local"
    state["ingested_at"] = datetime.now(timezone.utc).isoformat()
    _write_state(state)
    return state


def _write_state(state: Dict[str, Any]) -> None:
    _ensure_dirs()
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
