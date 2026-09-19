"""
Unit and integration test suite for the Agentic AI RAG system.
Tests:
- PDF text extraction & unicode normalization
- Document chunking and metadata preservation
- Local vector store cosine ranking and score boundary validation
- LangGraph RAG pipeline execution with conditional routing
- Out-of-scope query interception
- Groundedness check and self-correction
- FastAPI REST endpoints (/health, /chat, /ingest)
"""

import os
import re
import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
import numpy as np
from langchain_core.documents import Document
from fastapi.testclient import TestClient

import config
from document_session import save_uploaded_pdf, mark_ingested
from ingestion import (
    extract_text_from_pdf,
    chunk_documents,
    LocalVectorStore,
    get_embedding_model,
    ingest_pipeline,
    clean_text,
    extract_tables_and_media,
    table_to_markdown,
)
from rag_pipeline import (
    query_rag,
    build_rag_graph,
    check_grounding_node,
    correct_answer_node,
    is_answer_faithful,
    RAGState
)
from api import app


def test_clean_text():
    """Verify unicode normalization and ligature cleanup."""
    raw = "transfor-\nmative AI\u0562s systems \ufffd and\n\n\n\nmultiple lines"
    cleaned = clean_text(raw)
    assert "transformative" in cleaned
    assert "\ufffd" not in cleaned
    assert "\n\n\n" not in cleaned


def test_pdf_extraction():
    """Verify that PyMuPDF extracts clean pages from Ebook-Agentic-AI.pdf."""
    pages = extract_text_from_pdf(str(config.DEFAULT_PDF_PATH))
    assert len(pages) > 0, "Should extract at least one page from PDF"
    assert pages[0].metadata["page"] >= 1
    assert "source" in pages[0].metadata
    assert len(pages[0].page_content.strip()) > 10


def test_chunk_documents():
    """Verify that RecursiveCharacterTextSplitter generates well-formed chunks."""
    pages = extract_text_from_pdf(str(config.DEFAULT_PDF_PATH))
    chunks = chunk_documents(pages, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= len(pages), "Chunks count should be >= pages count"
    assert "page" in chunks[0].metadata
    assert "chunk_id" in chunks[0].metadata
    assert "index" in chunks[0].metadata


def test_local_vector_store_cosine_ranking_and_bounds():
    """
    Verify cosine similarity ranking and check that negative/low scores
    are bounded between 0.0 and 1.0 without inverted normalization bugs.
    """
    class MockEmbedding:
        def embed_documents(self, texts):
            # 2D dummy embeddings: doc1=[1, 0], doc2=[0, 1], doc3=[-1, 0]
            vectors = []
            for t in texts:
                if "positive" in t:
                    vectors.append([1.0, 0.0])
                elif "neutral" in t:
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([-1.0, 0.0])
            return vectors

        def embed_query(self, query):
            return [1.0, 0.0]

    mock_emb = MockEmbedding()
    store = LocalVectorStore(embedding_function=mock_emb)
    docs = [
        Document(page_content="positive match", metadata={"page": 1, "chunk_id": "p1_c1"}),
        Document(page_content="neutral orthogonal", metadata={"page": 2, "chunk_id": "p2_c1"}),
        Document(page_content="negative opposite", metadata={"page": 3, "chunk_id": "p3_c1"}),
    ]
    store.add_documents(docs)

    results = store.similarity_search_with_score("positive query", k=3)
    assert len(results) == 3

    doc0, score0 = results[0]
    doc1, score1 = results[1]
    doc2, score2 = results[2]

    # Positive match should have score 1.0
    assert doc0.page_content == "positive match"
    assert score0 == 1.0

    # Orthogonal match should have score 0.0
    assert doc1.page_content == "neutral orthogonal"
    assert score1 == 0.0

    # Opposite match should be clamped to 0.0 (NOT inverted to 0.40)
    assert doc2.page_content == "negative opposite"
    assert score2 == 0.0

    # Verify monotonicity: positive >= neutral >= negative
    assert score0 >= score1 >= score2


def test_langgraph_rag_workflow_grounded(extractive_only):
    """Verify LangGraph workflow returns required fields: answer, chunks, confidence_score."""
    result = query_rag("What is Agentic AI?", top_k=3)
    assert "question" in result
    assert "answer" in result
    assert "retrieved_chunks" in result
    assert "confidence_score" in result
    assert "grounded" in result

    assert len(result["answer"]) > 15
    assert 1 <= len(result["retrieved_chunks"]) <= 3
    assert 0.0 <= result["confidence_score"] <= 1.0
    assert result["grounded"] is True

    for chunk in result["retrieved_chunks"]:
        assert "page" in chunk
        assert "score" in chunk
        assert "text" in chunk


def test_langgraph_out_of_scope_routing(extractive_only):
    """Verify that an irrelevant question routes to out-of-scope handler via conditional edge."""
    result = query_rag("How do I bake a chocolate cake with sourdough starter?", top_k=2)
    assert result["confidence_score"] <= 0.30
    assert "does not contain" in result["answer"].lower()
    assert result["grounded"] is True
    assert result["correction_notes"] is not None


def test_langgraph_self_correction_branch():
    """Verify self-correction node activates when an ungrounded answer is detected."""
    mock_chunks = [
        {"chunk_id": "p1_c1", "page": 1, "score": 0.85, "text": "Agentic AI uses autonomous planning and tool execution."},
    ]
    # State with an answer talking about unrelated external knowledge without citing context
    state: RAGState = {
        "question": "What is Agentic AI?",
        "top_k": 1,
        "nvidia_api_key": None,
        "nvidia_model": None,
        "pinecone_api_key": None,
        "pinecone_index_name": None,
        "documents": [],
        "context_chunks": mock_chunks,
        "relevance_score": 0.85,
        "is_relevant": True,
        "answer": "Quantum superposition is when photons rotate in parallel dimensions across the universe.",
        "grounded": False,
        "confidence_score": 0.90,
        "correction_notes": None
    }

    grounding_check = check_grounding_node(state)
    assert grounding_check["grounded"] is False

    # Apply self-correction
    state.update(grounding_check)
    corrected_state = correct_answer_node(state)
    assert "Self-Correction Applied" in corrected_state["answer"]
    assert corrected_state["grounded"] is True
    assert corrected_state["confidence_score"] <= 0.70


def test_fastapi_health():
    """Verify FastAPI /health endpoint."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["document_ready"] is True
    assert data["vector_store_initialized"] is True
    assert data["indexed_chunks"] > 0


def test_fastapi_chat_endpoint(extractive_only):
    """Verify FastAPI /chat returns answer, retrieved_chunks, and confidence_score."""
    client = TestClient(app)
    payload = {
        "question": "What is an autonomous agent?",
        "top_k": 3
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Requirements from screenshot verification
    assert "answer" in data
    assert "retrieved_chunks" in data
    assert "confidence_score" in data
    assert 1 <= len(data["retrieved_chunks"]) <= 3
    assert 0.0 <= data["confidence_score"] <= 1.0


def test_fastapi_chat_empty_query():
    """Verify validation on empty query."""
    client = TestClient(app)
    response = client.post("/chat", json={"question": "   "})
    assert response.status_code == 400


def test_fastapi_ingest_endpoint():
    """Verify FastAPI /ingest endpoint."""
    client = TestClient(app)
    response = client.post("/ingest", json={"use_pinecone": False, "force_reindex": False, "vector_db_type": "local"})
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["total_chunks"] > 0


def test_nvidia_nim_llm():
    """Verify NVIDIA NIM Chat model invocation with the configured API key."""
    if not config.NVIDIA_API_KEY:
        pytest.skip("NVIDIA_API_KEY not configured")
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    llm = ChatNVIDIA(
        model=config.DEFAULT_NVIDIA_CHAT_MODEL,
        api_key=config.NVIDIA_API_KEY,
        temperature=0.0
    )
    res = llm.invoke("Say 'NVIDIA NIM active' in 3 words")
    assert res is not None
    assert len(res.content.strip()) > 0


def test_nvidia_embeddings():
    """Verify NVIDIA NIM Embeddings generation with the configured API key."""
    if not config.NVIDIA_API_KEY:
        pytest.skip("NVIDIA_API_KEY not configured")
    emb = get_embedding_model(provider="nvidia")
    vec = emb.embed_query("Agentic AI workflows")
    assert len(vec) == 2048
    assert isinstance(vec[0], float)


def test_chroma_embedded_vector_store():
    """Verify ChromaDB embedded serverless store, metadata persistence, and clean reset."""
    import tempfile
    from ingestion import store_in_chroma, connect_chroma
    emb = get_embedding_model(provider="local")
    test_docs = [
        Document(page_content="Agentic systems plan autonomously.", metadata={"page": 5, "chunk_id": "p5_c1"}),
        Document(page_content="Database architecture for embeddings.", metadata={"page": 12, "chunk_id": "p12_c1"}),
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        store = store_in_chroma(
            chunks=test_docs,
            embeddings=emb,
            persist_directory=tmp_path,
            collection_name="test_col",
            embedding_provider="local"
        )
        assert store._collection.count() == 2

        # Verify connect_chroma loads cleanly with metadata
        conn = connect_chroma(persist_directory=tmp_path, collection_name="test_col")
        assert conn is not None
        assert conn._collection.count() == 2

        results = conn.similarity_search_with_score("autonomous planning", k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert doc.metadata["page"] == 5

        # Verify re-ingest resets collection instead of duplicating
        re_store = store_in_chroma(
            chunks=test_docs,
            embeddings=emb,
            persist_directory=tmp_path,
            collection_name="test_col",
            embedding_provider="local"
        )
        assert re_store._collection.count() == 2
        re_conn = connect_chroma(persist_directory=tmp_path, collection_name="test_col")
        assert re_conn is not None
        assert re_conn._collection.count() == 2


def test_chroma_nvidia_embeddings_alignment():
    """Verify Chroma persistence and auto dimension alignment with NVIDIA embeddings (2048 dims)."""
    if not config.NVIDIA_API_KEY:
        pytest.skip("NVIDIA_API_KEY not configured")
    import tempfile
    from ingestion import store_in_chroma, connect_chroma
    nv_emb = get_embedding_model(provider="nvidia")
    test_docs = [
        Document(page_content="NVIDIA NIM accelerates enterprise generative AI workloads.", metadata={"page": 1, "chunk_id": "p1_c1"}),
        Document(page_content="Retrieval augmented generation connects LLMs to real-time facts.", metadata={"page": 2, "chunk_id": "p2_c1"}),
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        store_in_chroma(
            chunks=test_docs,
            embeddings=nv_emb,
            persist_directory=tmp_path,
            collection_name="test_nv_col",
            embedding_provider="nvidia"
        )

        # connect_chroma without passing embeddings: should auto-detect 2048-dim NVIDIA embeddings from metadata
        conn = connect_chroma(persist_directory=tmp_path, collection_name="test_nv_col")
        assert conn is not None
        assert conn._collection.count() == 2

        results = conn.similarity_search_with_score("enterprise generative AI", k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert doc.metadata["page"] == 1


def test_fastapi_chat_nvidia_endpoint():
    """Verify FastAPI /chat endpoint using NVIDIA NIM model."""
    if not config.NVIDIA_API_KEY:
        pytest.skip("NVIDIA_API_KEY not configured")
    client = TestClient(app)
    payload = {
        "question": "What is an AI Agent according to the ebook?",
        "top_k": 2,
        "nvidia_model": config.DEFAULT_NVIDIA_CHAT_MODEL
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 10
    assert "retrieved_chunks" in data
    assert 1 <= len(data["retrieved_chunks"]) <= 2
    assert data["confidence_score"] > 0.0


def test_nvidia_model_resilience_fallback():
    """Verify LangGraph query_rag gracefully handles deprecated or invalid NVIDIA models."""
    if not config.NVIDIA_API_KEY:
        pytest.skip("NVIDIA_API_KEY not configured")
    result = query_rag(
        question="What is Agentic AI?",
        top_k=2,
        nvidia_model="non-existent/model-that-does-not-exist"
    )
    assert "answer" in result
    assert len(result["answer"]) > 10
    assert result["grounded"] is True


def test_faithfulness_rejects_invented_numbers_and_facts():
    """Numeric and topical hallucinations must fail the faithfulness gate."""
    chunks = [{
        "chunk_id": "p13_c1",
        "page": 13,
        "score": 0.9,
        "text": "Automated task processing has reduced manual work by 40%, freeing teams to focus on strategic initiatives while reducing operational costs by 15%."
    }]
    invented = "Agentic AI reduced manual work by 90% and cut costs by 75% across every industry worldwide."
    assert is_answer_faithful(invented, chunks) is False

    grounded = "Automated task processing has reduced manual work by 40% [Page 13]."
    assert is_answer_faithful(grounded, chunks) is True

    oos = (
        "The uploaded PDF does not contain sufficient relevant information to answer this question."
    )
    assert is_answer_faithful(oos, chunks) is True
    mixed = oos + " Agentic AI will replace all human CEOs by 2027."
    assert is_answer_faithful(mixed, chunks) is False
    templated = "According to the ebook, Agentic AI teleports inventory across galaxies [Page 13]."
    assert is_answer_faithful(templated, chunks) is False


PDF_IN_SCOPE_CASES = [
    (
        "What is Agentic AI according to the ebook?",
        ["autonomous", "agentic", "decision"],
    ),
    (
        "How does Agentic AI differ from LLMs?",
        ["llm", "agent", "proactive", "reactive", "goal"],
    ),
    (
        "What are the core pillars of an Agentic AI system?",
        ["perception", "reasoning", "planning", "learning", "execution"],
    ),
    (
        "What operational efficiency benefits did the retail company see?",
        ["40%", "15%", "manual"],
    ),
    (
        "What healthcare use cases of Agentic AI are described?",
        ["patient", "treatment", "monitoring"],
    ),
    (
        "Who contributed to this Agentic AI executive guide?",
        ["konverge", "emergence"],
    ),
    (
        "What did McKinsey say about agents and review cycle times?",
        ["20", "60", "review"],
    ),
    (
        "How is Agentic AI different from traditional AI and RPA?",
        ["traditional", "rpa", "autonomous", "rule"],
    ),
]

PDF_OUT_OF_SCOPE_CASES = [
    "How do I bake a chocolate cake with sourdough starter?",
    "What is the current price of Bitcoin?",
    "Who won the 2018 FIFA World Cup?",
    "How do I reverse a linked list in Python?",
    "What is the weather in Tokyo tomorrow?",
]


def _assert_grounded_to_chunks(result: dict):
    answer = result["answer"]
    chunks = result["retrieved_chunks"]
    assert is_answer_faithful(answer, chunks) is True
    assert "**Citations:**" in answer or re.search(r"\[Page\s+\d+", answer)
    claim_nums = re.findall(r"\d+(?:\.\d+)?%", answer.split("**Citations:**")[0])
    if claim_nums and "does not contain sufficient" not in answer.lower():
        context = " ".join(c["text"] for c in chunks)
        for num in claim_nums:
            assert num in context, f"Hallucinated statistic {num} not in retrieved PDF chunks"


@pytest.mark.parametrize("question,must_any", PDF_IN_SCOPE_CASES, ids=[c[0][:40] for c in PDF_IN_SCOPE_CASES])
def test_pdf_in_scope_extractive_grounded(extractive_only, question, must_any):
    """Every in-scope ebook question must be answered from retrieved PDF text only."""
    result = query_rag(question, top_k=4, embedding_provider="local")
    answer_l = result["answer"].lower()
    assert "does not contain sufficient" not in answer_l
    assert result["grounded"] is True
    assert result["retrieved_chunks"]
    assert any(token.lower() in answer_l for token in must_any), (
        f"Answer missing expected ebook terms {must_any}: {result['answer'][:400]}"
    )
    _assert_grounded_to_chunks(result)


@pytest.mark.parametrize("question", PDF_OUT_OF_SCOPE_CASES)
def test_pdf_out_of_scope_refuses(extractive_only, question):
    """Questions outside the ebook must refuse instead of hallucinating."""
    result = query_rag(question, top_k=4, embedding_provider="local")
    assert "does not contain sufficient" in result["answer"].lower()
    assert result["grounded"] is True
    assert result["confidence_score"] <= 0.30


@pytest.mark.parametrize("question,must_any", PDF_IN_SCOPE_CASES, ids=[f"nv-{c[0][:32]}" for c in PDF_IN_SCOPE_CASES])
def test_pdf_in_scope_nvidia_no_hallucination(question, must_any):
    """Live NVIDIA answers must stay faithful to retrieved PDF chunks."""
    if not config.NVIDIA_API_KEY:
        pytest.skip("NVIDIA_API_KEY not configured")
    result = query_rag(
        question,
        top_k=4,
        embedding_provider="local",
        nvidia_model=config.DEFAULT_NVIDIA_CHAT_MODEL,
    )
    answer_l = result["answer"].lower()
    assert result["grounded"] is True
    assert "does not contain sufficient" not in answer_l
    assert any(token.lower() in answer_l for token in must_any), (
        f"NVIDIA answer missing expected ebook terms {must_any}: {result['answer'][:500]}"
    )
    _assert_grounded_to_chunks(result)


@pytest.mark.parametrize("question", PDF_OUT_OF_SCOPE_CASES, ids=[f"nv-oos-{i}" for i in range(len(PDF_OUT_OF_SCOPE_CASES))])
def test_pdf_out_of_scope_nvidia_refuses(question):
    """Live NVIDIA path must refuse out-of-ebook questions."""
    if not config.NVIDIA_API_KEY:
        pytest.skip("NVIDIA_API_KEY not configured")
    result = query_rag(
        question,
        top_k=4,
        embedding_provider="local",
        nvidia_model=config.DEFAULT_NVIDIA_CHAT_MODEL,
    )
    assert "does not contain sufficient" in result["answer"].lower()
    assert result["grounded"] is True


def test_pdf_tables_extracted_as_markdown():
    """PDF comparison tables must be recovered as markdown, not flattened prose."""
    tables, _images, manifest = extract_tables_and_media(str(config.DEFAULT_PDF_PATH))
    assert tables, "Expected at least one real table in Ebook-Agentic-AI.pdf"
    joined = "\n".join(t.page_content for t in tables)
    assert "Traditional AI" in joined
    assert "Agentic AI" in joined
    assert any("| --- |" in (t.metadata.get("table_markdown") or "") for t in tables)
    assert manifest.get("tables")


def test_table_question_returns_markdown_table(extractive_only):
    """Comparison questions should answer with a cited markdown table from the PDF."""
    result = query_rag(
        "Compare Traditional AI, Non-agentic AI, Agentic AI and Generative AI in a table",
        top_k=6,
        embedding_provider="local",
    )
    answer = result["answer"]
    assert result["grounded"] is True
    assert "does not contain sufficient" not in answer.lower()
    assert "|" in answer
    assert "Traditional AI" in answer
    assert "Agentic AI" in answer
    assert re.search(r"\[Page\s+\d+", answer)
    assert "**Citations:**" in answer
    assert result.get("tables"), "Expected structured table payload"


def test_image_question_retrieves_pdf_figure(extractive_only):
    """Visual questions should retrieve an on-disk figure from the PDF page."""
    result = query_rag(
        "Show the diagram comparing LLMs and Agents with reinforcement learning and neural networks",
        top_k=6,
        embedding_provider="local",
    )
    assert result["grounded"] is True
    images = result.get("images") or []
    assert images, f"Expected retrieved figures, answer was: {result['answer'][:400]}"
    existing = [img for img in images if img.get("path") and Path(img["path"]).exists()]
    assert existing, f"Figure files missing: {images}"
    assert any(img.get("page") for img in existing)
    assert re.search(r"\[Page\s+\d+", result["answer"])


def test_every_in_scope_answer_has_page_citations(extractive_only):
    """Grounded answers must cite PDF pages."""
    result = query_rag("What is Agentic AI according to the ebook?", top_k=4, embedding_provider="local")
    assert re.search(r"\[Page\s+\d+", result["answer"])
    assert "**Citations:**" in result["answer"]


def test_table_to_markdown_roundtrip():
    rows = [
        ["Type", "Definition"],
        ["Agentic AI", "Autonomous decision-making"],
        ["Generative AI", "Creates new content"],
    ]
    md = table_to_markdown(rows)
    assert md.splitlines()[0].startswith("| Type |")
    assert "Agentic AI" in md
    assert "---" in md


def test_streamlit_config_file_watcher():
    """Verify Streamlit config disables fileWatcherType to prevent deep scans."""
    config_file = ROOT_DIR / ".streamlit" / "config.toml"
    assert config_file.exists(), ".streamlit/config.toml should exist"
    content = config_file.read_text(encoding="utf-8")
    assert 'fileWatcherType = "none"' in content or "fileWatcherType = 'none'" in content


def test_lazy_loading_prevents_transformers_import_on_startup():
    """Verify that importing ingestion does not eagerly load transformers or sentence_transformers."""
    import subprocess
    code = (
        "import sys, ingestion; "
        "has_trans = 'transformers' in sys.modules; "
        "has_st = 'sentence_transformers' in sys.modules; "
        "sys.exit(0 if (not has_trans and not has_st) else 1)"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert res.returncode == 0, f"Importing ingestion unexpectedly loaded transformers/sentence-transformers: {res.stderr.decode()}"


def test_torchvision_dynamic_stub_fallback_robustness():
    """
    Verify that the dynamic meta-path finder fallback safely handles:
    - Missing torchvision module and nested subpackage imports
    - Enum / attribute lookups (e.g. InterpolationMode.NEAREST_EXACT)
    - Python 3.10+ union type expressions (EnumType | tvF.InterpolationMode)
    - Subclassing stub classes
    - Streamlit LocalSourcesWatcher path extraction without errors
    """
    import subprocess
    code = """
import sys, enum, typing
from importlib.machinery import PathFinder

# 1. Mask real torchvision on disk so Python treats it as uninstalled
class MaskTorchvision(PathFinder):
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname == "torchvision" or fullname.startswith("torchvision."):
            return None
        return super().find_spec(fullname, path, target)

sys.meta_path = [MaskTorchvision if f is PathFinder else f for f in sys.meta_path]

# 2. Clean out any pre-loaded torchvision modules
for m in list(sys.modules):
    if "torchvision" in m:
        del sys.modules[m]

# 3. Trigger _ensure_torchvision_available to install fallback finder
from ingestion import _ensure_torchvision_available
_ensure_torchvision_available()

# 4. Verify subpackage import
from torchvision.transforms.v2 import functional as tvF
from torchvision.transforms import InterpolationMode

# 5. Verify attribute access
val = InterpolationMode.NEAREST_EXACT
assert val is not None

# 6. Verify union typing support with Enums (Python 3.10+ __ror__ / __or__)
class SampleEnum(enum.Enum):
    VAL = 1

union_type = SampleEnum | tvF.InterpolationMode | int | None
assert union_type is not None

# 7. Verify subclassing
class CustomProcessor(tvF.SomeBackend):
    pass
assert issubclass(CustomProcessor, object)

# 8. Verify Streamlit LocalSourcesWatcher inspection
from streamlit.watcher.local_sources_watcher import get_module_paths
import torchvision
paths = get_module_paths(torchvision)
assert isinstance(paths, set)
print("STUB_VERIFICATION_PASSED")
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Stub fallback verification failed: {res.stderr}"
    assert "STUB_VERIFICATION_PASSED" in res.stdout



def test_torchvision_resolution_and_watcher_stability():
    """Verify torchvision imports succeed and LocalSourcesWatcher get_module_paths does not crash."""
    from ingestion import _ensure_torchvision_available
    _ensure_torchvision_available()
    import torchvision
    import torchvision.transforms.v2
    import torchvision.io
    assert torchvision is not None

    from streamlit.watcher.local_sources_watcher import get_module_paths
    for name, mod in list(sys.modules.items()):
        if name.startswith("transformers"):
            paths = get_module_paths(mod)
            assert isinstance(paths, set)

