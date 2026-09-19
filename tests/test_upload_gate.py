"""
BUG-R upload gate regressions.

Without an uploaded PDF the system must refuse to chat (API and retrieval).
Upload + ingest uses the same helpers as Streamlit so UI/API cannot drift.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from fastapi.testclient import TestClient

import config
import document_session as ds
from api import app
from ingestion import set_active_vector_store, get_active_vector_store, ingest_pipeline


def test_chat_blocked_without_upload_bug_r_upload_gate(without_document):
    """BUG-R1: /chat must not answer if no PDF was uploaded."""
    client = TestClient(app)
    response = client.post("/chat", json={"question": "What is Agentic AI?", "top_k": 2})
    assert response.status_code == 409
    assert "upload" in response.json()["detail"].lower()


def test_ingest_blocked_without_upload_bug_r_ingest_gate(without_document):
    """BUG-R2: /ingest must not fall back to a bundled default PDF."""
    client = TestClient(app)
    response = client.post("/ingest", json={"force_reindex": False, "use_pinecone": False})
    assert response.status_code == 409
    assert "upload" in response.json()["detail"].lower()


def test_health_reports_not_ready_without_upload(without_document):
    client = TestClient(app)
    data = client.get("/health").json()
    assert data["document_ready"] is False
    assert data["indexed_chunks"] == 0


def test_document_status_endpoint_without_upload(without_document):
    client = TestClient(app)
    data = client.get("/document").json()
    assert data["ready"] is False
    assert data["document"] is None


def test_upload_rejects_non_pdf(without_document):
    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400
    assert "pdf" in response.json()["detail"].lower()


def test_upload_rejects_empty_file(without_document):
    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400


def test_upload_rejects_spoofed_extension(without_document):
    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"file": ("fake.pdf", b"PK\x03\x04this is zip", "application/pdf")},
    )
    assert response.status_code == 400
    assert "valid pdf" in response.json()["detail"].lower()


def test_upload_then_chat_parity_with_ui_path(without_document, monkeypatch):
    """BUG-R3: API upload must use the same ingest path Streamlit uses."""
    from rag_pipeline import query_rag

    monkeypatch.setattr("rag_pipeline.get_llm", lambda **kwargs: None)
    pdf_bytes = Path(config.DEFAULT_PDF_PATH).read_bytes()
    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"file": ("Ebook-Agentic-AI.pdf", pdf_bytes, "application/pdf")},
        data={"embedding_provider": "local", "vector_db_type": "local"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_chunks"] > 0
    assert body["filename"] == "Ebook-Agentic-AI.pdf"
    assert ds.is_document_ready() is True

    chat = client.post("/chat", json={"question": "What is Agentic AI?", "top_k": 2})
    assert chat.status_code == 200
    data = chat.json()
    assert "answer" in data
    assert len(data["answer"]) > 10
    assert data.get("citations") or "[Page" in data["answer"]

    # Direct pipeline (Streamlit) after the same upload
    result = query_rag("What is Agentic AI?", top_k=2, embedding_provider="local")
    assert result["grounded"] is True
    assert result["answer"]


def test_save_uploaded_pdf_helper_matches_api_rules(without_document):
    with pytest.raises(ValueError):
        ds.save_uploaded_pdf("file.docx", b"%PDF-1.4 dummy")
    with pytest.raises(ValueError):
        ds.save_uploaded_pdf("file.pdf", b"")
    state = ds.save_uploaded_pdf("My Report.PDF", b"%PDF-1.7\n% test")
    assert state["ingested"] is False
    assert Path(state["path"]).is_file()
    assert state["filename"] == "My Report.PDF"
    assert ds.is_document_ready() is False


def test_query_rag_blocked_without_upload(without_document):
    """BUG-R5: retrieval engine must refuse, not only the HTTP /chat wrapper."""
    from rag_pipeline import query_rag
    with pytest.raises(ds.DocumentNotReady):
        query_rag("What is Agentic AI?", top_k=2, embedding_provider="local")


def test_leftover_vector_store_ignored_without_upload(without_document):
    """BUG-R6: a leftover in-memory store must not answer after the PDF is removed."""
    from langchain_core.documents import Document
    from ingestion import LocalVectorStore, get_embedding_model, get_active_vector_store
    dummy = LocalVectorStore(get_embedding_model(provider="local"))
    dummy.add_documents([
        Document(page_content="secret leftover corpus about penguins", metadata={"page": 1, "source": "leak.pdf"})
    ])
    set_active_vector_store(dummy)
    assert ds.is_document_ready() is False
    assert get_active_vector_store() is None
    with pytest.raises(ds.DocumentNotReady):
        from rag_pipeline import query_rag
        query_rag("penguins", top_k=1, embedding_provider="local")


def test_active_json_cannot_point_at_bundled_ebook(without_document):
    """BUG-R7: active.json must not treat the repo ebook as an uploaded corpus."""
    ds._write_state({
        "path": str(config.DEFAULT_PDF_PATH.resolve()),
        "ingested": True,
        "filename": "Ebook-Agentic-AI.pdf",
    })
    assert ds.get_active_document() is None
    assert ds.is_document_ready() is False


def test_ingest_ignores_client_pdf_path(without_document, monkeypatch):
    """BUG-R8: /ingest always uses the uploaded path, never payload.pdf_path."""
    monkeypatch.setattr("rag_pipeline.get_llm", lambda **kwargs: None)
    pdf_bytes = Path(config.DEFAULT_PDF_PATH).read_bytes()
    client = TestClient(app)
    up = client.post(
        "/upload",
        files={"file": ("Ebook-Agentic-AI.pdf", pdf_bytes, "application/pdf")},
        data={"embedding_provider": "local", "vector_db_type": "local"},
    )
    assert up.status_code == 200
    active = ds.get_active_document()
    captured = {}

    def spy(**kwargs):
        captured.update(kwargs)
        return ingest_pipeline(**kwargs)

    monkeypatch.setattr("api.ingest_pipeline", spy)
    res = client.post(
        "/ingest",
        json={"pdf_path": str(config.DEFAULT_PDF_PATH), "force_reindex": False, "use_pinecone": False},
    )
    assert res.status_code == 200
    assert captured["pdf_path"] == active["path"]
    assert ds.UPLOAD_DIR.resolve() in Path(captured["pdf_path"]).resolve().parents


def test_upload_filename_cannot_escape_upload_dir(without_document):
    """BUG-R4: Path-traversal names must stay inside .uploads."""
    state = ds.save_uploaded_pdf("..\\..\\etc\\passwd.pdf", b"%PDF-1.4\n")
    stored = Path(state["path"]).resolve()
    assert ds.UPLOAD_DIR.resolve() in stored.parents
    assert stored.name.endswith(".pdf")
