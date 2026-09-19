import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest

import config
import document_session as ds
from document_session import save_uploaded_pdf, mark_ingested
from ingestion import ingest_pipeline, set_active_vector_store, get_active_vector_store


@pytest.fixture(scope="session", autouse=True)
def setup_ingestion():
    pdf_bytes = Path(config.DEFAULT_PDF_PATH).read_bytes()
    state = save_uploaded_pdf(Path(config.DEFAULT_PDF_PATH).name, pdf_bytes)
    res = ingest_pipeline(
        pdf_path=state["path"],
        embedding_provider="local",
        use_pinecone=False,
        force_reindex=False,
    )
    mark_ingested(res["total_pages"], res["total_chunks"], res.get("target_store", ""))


@pytest.fixture
def without_document():
    prev_state = ds.get_active_document()
    prev_store = get_active_vector_store(embedding_provider="local")
    ds.clear_active_document()
    set_active_vector_store(None)
    yield
    if prev_state:
        ds._write_state(prev_state)
        if prev_store is not None:
            set_active_vector_store(prev_store)
        else:
            res = ingest_pipeline(
                pdf_path=prev_state["path"],
                embedding_provider="local",
                use_pinecone=False,
                force_reindex=False,
            )
            ds.mark_ingested(res["total_pages"], res["total_chunks"], res.get("target_store", ""))
    else:
        set_active_vector_store(prev_store)


@pytest.fixture
def extractive_only(monkeypatch):
    monkeypatch.setattr("rag_pipeline.get_llm", lambda **kwargs: None)
