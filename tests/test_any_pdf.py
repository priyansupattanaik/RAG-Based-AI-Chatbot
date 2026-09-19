"""Grounding and citation tests on arbitrary PDFs, plus failure cases."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pymupdf
import pytest

import document_session as ds
from ingestion import ingest_pipeline, set_active_vector_store
from rag_pipeline import (
    query_rag,
    sanitize_page_citations,
    citation_pages_in_answer,
    retrieved_pages,
    OUT_OF_SCOPE_ANSWER,
)


def _make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def _index_bytes(name: str, data: bytes):
    state = ds.save_uploaded_pdf(name, data)
    set_active_vector_store(None)
    res = ingest_pipeline(
        pdf_path=state["path"],
        embedding_provider="local",
        use_pinecone=False,
        force_reindex=True,
    )
    ds.mark_ingested(res["total_pages"], res["total_chunks"], res.get("target_store", ""))
    return state, res


def test_sanitize_drops_pages_that_were_not_retrieved():
    chunks = [{"page": 3, "text": "copper kettle 1842 Lisbon"}]
    raw = "The kettle is from 1842 [Page 3] and also [Page 99] (Page 88) Page 77 according to Wikipedia."
    cleaned = sanitize_page_citations(raw, chunks)
    assert "[Page 3]" in cleaned
    assert "[Page 99]" not in cleaned
    assert "Page 88" not in cleaned
    assert "Page 77" not in cleaned


def test_citation_pages_helper():
    pages = citation_pages_in_answer("See [Page 2] and [Page 10].")
    assert pages == {2, 10}


def test_any_pdf_answers_from_correct_page(without_document, extractive_only):
    data = _make_pdf([
        "The copper kettle was invented in 1842 in Lisbon.",
        "Cats are fed at 06:00 every morning in the annex.",
    ])
    _index_bytes("kettle-notes.pdf", data)

    kettle = query_rag("When was the copper kettle invented?", top_k=3, embedding_provider="local")
    assert kettle["grounded"] is True
    assert "1842" in kettle["answer"]
    assert "does not contain sufficient" not in kettle["answer"].lower()
    cited = citation_pages_in_answer(kettle["answer"])
    allowed = retrieved_pages(kettle["retrieved_chunks"])
    assert cited
    assert cited <= allowed
    assert 1 in cited

    cats = query_rag("When are the cats fed?", top_k=3, embedding_provider="local")
    assert "06:00" in cats["answer"]
    cat_cited = citation_pages_in_answer(cats["answer"])
    assert cat_cited
    assert cat_cited <= retrieved_pages(cats["retrieved_chunks"])


def test_any_pdf_refuses_outside_facts(without_document, extractive_only):
    data = _make_pdf(["The copper kettle was invented in 1842 in Lisbon."])
    _index_bytes("kettle-only.pdf", data)
    result = query_rag("What is Agentic AI?", top_k=3, embedding_provider="local")
    assert "does not contain sufficient" in result["answer"].lower()
    assert "autonomous multi-agent" not in result["answer"].lower()


def test_empty_pdf_does_not_invent_facts(without_document, extractive_only):
    data = _make_pdf([" "])
    _index_bytes("blank.pdf", data)
    result = query_rag("What is the capital of France?", top_k=2, embedding_provider="local")
    assert "paris" not in result["answer"].lower()
    assert "does not contain sufficient" in result["answer"].lower()


def test_answer_citations_are_subset_of_retrieved_chunks(extractive_only):
    result = query_rag("What is Agentic AI according to the ebook?", top_k=4, embedding_provider="local")
    allowed = retrieved_pages(result["retrieved_chunks"])
    cited = citation_pages_in_answer(result["answer"])
    assert cited
    assert cited <= allowed


def test_whitespace_question_is_rejected_by_api():
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app)
    response = client.post("/chat", json={"question": "   ", "top_k": 2})
    assert response.status_code == 400


def test_hallucinated_claim_numbers_substring_leak():
    from rag_pipeline import is_answer_faithful
    chunks = [{
        "text": "In 2024, the company launched 100 satellites for space communication missions successfully."
    }]
    # Hallucinated number 20 (substring of 2024)
    ans20 = "The company launched 20 satellites for space communication missions successfully."
    assert is_answer_faithful(ans20, chunks) is False

    # Hallucinated number 1 (substring of 100)
    ans1 = "The company launched 1 satellite for space communication missions successfully."
    assert is_answer_faithful(ans1, chunks) is False


def test_hallucinated_slip_in_sentence_rejected():
    from rag_pipeline import is_answer_faithful
    chunks = [{
        "text": "Sentence one about science. Sentence two about biology. Sentence three about chemistry. "
                "Sentence four about geology. Sentence five about physics. Sentence six about astronomy. "
                "Sentence seven about ecology."
    }]
    # 7 grounded sentences + 1 completely fabricated sentence
    ans = (
        "Sentence one about science. Sentence two about biology. Sentence three about chemistry. "
        "Sentence four about geology. Sentence five about physics. Sentence six about astronomy. "
        "Sentence seven about ecology. Aliens built the pyramids in Egypt."
    )
    assert is_answer_faithful(ans, chunks) is False


def test_partial_overlap_out_of_scope_query_refuses(extractive_only):
    # Questions that have domain words ('Agent') but ask for out-of-document facts
    pasta = query_rag("Can an Agent cook pasta?", top_k=4, embedding_provider="local")
    assert "does not contain sufficient" in pasta["answer"].lower()
    assert pasta["confidence_score"] <= 0.30

    salary = query_rag("What is the salary of an Agentic AI engineer in 2025?", top_k=4, embedding_provider="local")
    assert "does not contain sufficient" in salary["answer"].lower()
    assert salary["confidence_score"] <= 0.30


def test_multilingual_unicode_pdf(without_document, extractive_only):
    data = _make_pdf([
        "L'énergie solaire photovoltaïque est renouvelable et efficace à 95% pour l'industrie moderne.",
    ])
    _index_bytes("energie-solaire.pdf", data)
    res = query_rag("Quel est le taux d'efficacité de l'énergie solaire?", top_k=3, embedding_provider="local")
    assert res["grounded"] is True
    assert "95%" in res["answer"]
    assert "does not contain sufficient" not in res["answer"].lower()


def test_citing_non_retrieved_page_invalidates_grounding():
    from rag_pipeline import check_grounding_node
    state = {
        "answer": "Autonomous agents plan workflows [Page 99].",
        "context_chunks": [{"page": 1, "text": "Autonomous agents plan workflows."}],
        "relevance_score": 0.85,
    }
    res = check_grounding_node(state)
    assert res["grounded"] is False

