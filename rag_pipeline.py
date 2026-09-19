"""Retrieve PDF chunks, answer only from those chunks, cite their page numbers."""

import math
import re
import logging
from typing import TypedDict, List, Dict, Any, Optional, Set

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

import config
from ingestion import get_active_vector_store, load_asset_manifest

logger = logging.getLogger("rag_pipeline")
logging.basicConfig(level=logging.INFO)

OUT_OF_SCOPE_ANSWER = (
    "The uploaded PDF does not contain sufficient relevant information to answer this question."
)

_PAGE_CITE = re.compile(r"\[Page\s+(\d+)\]", re.IGNORECASE)

_STOPWORDS: Set[str] = {
    "about", "after", "also", "been", "being", "does", "from", "have", "into",
    "just", "like", "more", "most", "only", "other", "over", "some", "such",
    "than", "that", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "under", "very", "what", "when", "where", "which",
    "while", "with", "would", "your", "the", "and", "for", "are", "but",
    "not", "you", "all", "can", "her", "was", "one", "our", "out", "how",
    "who", "why", "did", "its", "has", "had", "his", "she", "any", "few",
    "according", "page", "score", "extracted", "directly", "referenced",
    "described",
}


def _stem(token: str) -> str:
    for suffix in ("tions", "tion", "ments", "ment", "ting", "ted", "ing", "ers", "ies", "ied", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _content_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[^\W\d_][\w%]{1,}", (text or "").lower())
    return [_stem(t) for t in tokens if t not in _STOPWORDS]


def _query_terms(question: str) -> List[str]:
    return _content_tokens(question) + re.findall(r"\d+(?:\.\d+)?%?", question or "")


def _lexical_overlap(question: str, text: str) -> float:
    q = set(_query_terms(question))
    if not q:
        return 0.0
    text_l = (text or "").lower()
    t = set(_content_tokens(text)) | set(re.findall(r"\d+(?:\.\d+)?%?", text or ""))
    hits = 0
    for term in q:
        if term in t or term.lower() in text_l:
            hits += 1
    return hits / len(q)


def _extract_claim_numbers(text: str) -> List[str]:
    """Numbers that are factual claims, excluding citations and retrieval scores."""
    stripped = re.sub(r"\[?Pages?\s+[\d,\s\-and]+\]?", " ", text, flags=re.IGNORECASE)
    stripped = re.sub(r"\(Pages?\s+[\d,\s\-and]+\)", " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\bPages?\s+\d+\b", " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"similarity score\s+[0-9.]+", " ", stripped, flags=re.IGNORECASE)
    raw = re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?%?\b", stripped)
    return [n.replace(",", "") for n in raw]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _is_refusal(answer: str) -> bool:
    compact = re.sub(r"\s+", " ", (answer or "").strip().lower())
    canned = re.sub(r"\s+", " ", OUT_OF_SCOPE_ANSWER.strip().lower())
    if compact == canned:
        return True
    if compact.startswith(canned) and len(compact) < len(canned) + 8:
        return True
    return False


def is_answer_faithful(answer: str, chunks: List[Dict[str, Any]]) -> bool:
    """
    True only if the answer is supported by retrieved chunks:
    - exact out-of-scope refusals are faithful
    - every claim number must appear in the retrieved text
    - every substantive sentence must be a context span or high token overlap
    """
    if not answer:
        return False
    if _is_refusal(answer):
        return True
    if "does not contain sufficient" in answer.lower():
        return False
    if not chunks:
        return False
    if "**Citations:**" in answer:
        answer = answer.split("**Citations:**")[0]
    answer = re.sub(r"\nSources:\s*\[Page[\s\S]*$", "", answer, flags=re.IGNORECASE)

    context = " ".join(c.get("text", "") for c in chunks)
    context_norm = _normalize_ws(context)
    context_nums = set(_extract_claim_numbers(context))
    context_tokens = set(_content_tokens(context))

    for num in _extract_claim_numbers(answer):
        norm_num = num.replace(",", "")
        if norm_num not in context_nums:
            return False

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 20]
    if not sentences:
        tokens = _content_tokens(answer)
        if not tokens:
            return False
        return all(t in context_tokens for t in tokens)

    for sentence in sentences:
        sl = sentence.lower()
        if "self-correction applied" in sl or "extracted directly from referenced" in sl:
            continue
        if sl.startswith("**from [page") or sl.startswith("from [page"):
            continue
        if re.search(r"page\s+\d+", sl):
            if sl.startswith("from the uploaded pdf"):
                continue
            if sl.startswith(("according to", "from the uploaded", "based on", "as stated in")):
                from document_session import get_active_document
                active_doc = get_active_document() or {}
                doc_terms = set(_content_tokens(active_doc.get("filename", "")))
                preamble_tokens = set(_content_tokens(sentence)) - {
                    "document", "pdf", "file", "text", "source", "uploaded", "ebook", "reference", "page"
                } - doc_terms
                if not preamble_tokens:
                    continue
        if _normalize_ws(sentence) in context_norm:
            continue
        tokens = _content_tokens(sentence)
        if len(tokens) < 3:
            continue
        if all(t in context_tokens for t in tokens):
            continue
        overlap = sum(1 for t in tokens if t in context_tokens)
        if (overlap / len(tokens)) >= 0.70:
            continue
        # Unsubstantiated sentence detected
        return False
    return True


def _lexical_search_store(store, question: str, k: int) -> List[tuple]:
    """Keyword/number overlap over every local document (BM25-style sparse retrieval)."""
    docs = getattr(store, "documents", None)
    if not docs:
        return []
    scored = []
    terms = [t for t in _query_terms(question) if len(t) >= 4]
    nums = re.findall(r"\d+(?:\.\d+)?%?", question)
    n_docs = max(len(docs), 1)
    df = {
        t: sum(1 for d in docs if t in d.page_content.lower())
        for t in terms
    }
    for doc in docs:
        text = doc.page_content
        text_l = text.lower()
        lex = _lexical_overlap(question, text)
        idf_bonus = 0.0
        for t in terms:
            if t in text_l:
                idf_bonus += 0.15 * math.log((n_docs + 1) / (df.get(t, 0) + 1))
        idf_bonus += 0.3 * sum(1 for n in nums if n in text)
        scored.append((doc, min(1.0, lex + idf_bonus)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [(doc, score) for doc, score in scored[:k] if score > 0]


def _rrf_fuse(dense: List[tuple], sparse: List[tuple], k: int) -> List[tuple]:
    """Reciprocal Rank Fusion of dense cosine hits and lexical hits."""
    fused: Dict[str, Dict[str, Any]] = {}

    def _key(doc: Document) -> str:
        return str(doc.metadata.get("chunk_id") or "") + "::" + doc.page_content[:96]

    def _add(pairs: List[tuple], weight: float, score_field: str):
        for rank, (doc, score) in enumerate(pairs):
            entry = fused.setdefault(_key(doc), {"doc": doc, "rrf": 0.0, "dense": 0.0, "lex": 0.0})
            entry["rrf"] += weight / (60 + rank)
            entry[score_field] = max(entry[score_field], float(score))

    _add(dense, 0.65, "dense")
    _add(sparse, 0.35, "lex")

    ranked = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
    out = []
    for entry in ranked[:k]:
        if entry["dense"] > 0 and entry["lex"] > 0:
            combined = min(1.0, 0.6 * entry["dense"] + 0.4 * entry["lex"])
        else:
            combined = max(entry["dense"], entry["lex"])
        out.append((entry["doc"], round(combined, 4)))
    return out


def _expand_same_page_chunks(store, results: List[tuple], max_per_page: int = 3) -> List[tuple]:
    """Pull sibling chunks from the same PDF page so headings are not retrieved without their body."""
    docs = getattr(store, "documents", None)
    if not docs or not results:
        return results
    pages = []
    seen_pages = set()
    for doc, _score in results[:6]:
        page = doc.metadata.get("page")
        if page not in seen_pages:
            seen_pages.add(page)
            pages.append(page)
    existing = {id(doc) for doc, _ in results}
    extras = []
    for page in pages:
        page_docs = [d for d in docs if d.metadata.get("page") == page]
        special = [d for d in page_docs if d.metadata.get("content_type") in {"table", "image"}]
        rest = [d for d in page_docs if d.metadata.get("content_type") not in {"table", "image"}][:max_per_page]
        for doc in special + rest:
            if id(doc) not in existing:
                extras.append((doc, 0.62))
                existing.add(id(doc))
    return results + extras


def _missing_proper_nouns(question: str, chunks: List[Dict[str, Any]]) -> List[str]:
    proper = [w for w in re.findall(r"\b[\w]{3,}\b", question or "") if w[0].isupper()]
    skip = {
        "What", "When", "Where", "Which", "This", "That", "These", "Those",
        "Compare", "Show", "Describe", "Explain", "List", "Give", "Tell", "Does",
        "How", "Who", "Whom", "Whose", "Why", "Please", "Could", "Would", "Should",
        "From", "With", "About", "Into", "Under", "Over", "Between", "Among",
        "Have", "Having", "Been", "Being", "Were", "Will", "State", "Define",
        "Find", "Detail", "Outline", "Summarize", "Identify", "Provide", "Can",
        "According", "Based", "Table", "Figure", "Check", "Also", "Make", "Name",
        "Is", "Are", "Was", "Do", "Did", "May", "Might", "Must", "Shall",
        "In", "On", "At", "By", "For", "To", "If", "As", "Quel", "Quelle",
        "Quels", "Quelles", "Comment", "Pourquoi", "Combien", "Quand", "Cual",
    }
    from document_session import get_active_document
    active_doc = get_active_document() or {}
    for term in re.findall(r"[\w]{3,}", active_doc.get("filename", "")):
        skip.add(term)
        skip.add(term.capitalize())
        skip.add(term.upper())
    context = " ".join(c.get("text", "") for c in chunks).lower()
    missing = []
    for noun in proper:
        if noun in skip:
            continue
        if noun.lower() not in context and _stem(noun.lower()) not in context:
            missing.append(noun)
    return missing


def _get_corpus_frequent_terms(chunks: List[Dict[str, Any]], threshold_ratio: float = 0.60) -> Set[str]:
    if not chunks:
        return set()
    chunk_token_sets = [set(_content_tokens(c.get("text", ""))) for c in chunks]
    all_terms = set().union(*chunk_token_sets) if chunk_token_sets else set()
    corpus_terms = set()
    min_count = max(2, int(len(chunks) * threshold_ratio))
    for t in all_terms:
        if sum(1 for cts in chunk_token_sets if t in cts) >= min_count:
            corpus_terms.add(t)
    return corpus_terms


def _distinctive_inquiry_terms(question: str, chunks: List[Dict[str, Any]]) -> Set[str]:
    meta_terms = {
        "document", "pdf", "page", "section", "chapter", "text", "file",
        "information", "content", "mention", "mentioned", "describe", "described",
        "state", "stated", "detail", "details", "explain", "explained",
        "according", "ebook", "guide", "paper", "book", "report",
    }
    from document_session import get_active_document
    active_doc = get_active_document() or {}
    doc_terms = set(_content_tokens(active_doc.get("filename", "")))
    corpus_terms = _get_corpus_frequent_terms(chunks)
    generic = meta_terms | doc_terms | corpus_terms
    q_tokens = set(_query_terms(question))
    return {t for t in q_tokens - generic if len(t) >= 4}


def _is_table_question(question: str) -> bool:
    q = (question or "").lower()
    hints = (
        "table", "compare", "comparison", "versus", " vs ", "types of",
        "capabilities", "difference between", "markdown table", "in a table",
        "columns", "rows"
    )
    return any(h in q for h in hints)


def _is_image_question(question: str) -> bool:
    q = (question or "").lower()
    hints = (
        "image", "figure", "diagram", "picture", "illustration", "show me",
        "visual", "infographic", "graphic", "photo", "screenshot"
    )
    return any(h in q for h in hints)


def collect_tables(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tables = []
    seen = set()
    for chunk in chunks:
        md = chunk.get("table_markdown")
        if not md and (chunk.get("content_type") == "table") and "|" in (chunk.get("text") or ""):
            parts = chunk["text"].split("\n\n", 1)
            md = parts[-1] if parts else chunk["text"]
        if not md or md in seen:
            continue
        seen.add(md)
        tables.append({
            "page": chunk.get("page"),
            "chunk_id": chunk.get("chunk_id"),
            "caption": chunk.get("caption") or f"Table from page {chunk.get('page')}",
            "markdown": md,
        })
    if tables:
        return tables
    manifest = load_asset_manifest()
    pages = {c.get("page") for c in chunks}
    for item in manifest.get("tables", []):
        if item.get("page") in pages and item.get("markdown") not in seen:
            seen.add(item["markdown"])
            tables.append(item)
    return tables


def collect_images(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    images = []
    seen = set()
    for chunk in chunks:
        path = chunk.get("image_path")
        if not path:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        images.append({
            "page": chunk.get("page"),
            "chunk_id": chunk.get("chunk_id"),
            "caption": chunk.get("caption") or f"Figure from page {chunk.get('page')}",
            "path": path,
            "url": chunk.get("image_url"),
        })
    manifest = load_asset_manifest()
    pages = {c.get("page") for c in chunks}
    for item in manifest.get("images", []):
        if item.get("page") not in pages:
            continue
        key = str(item.get("path"))
        if key in seen:
            continue
        seen.add(key)
        images.append(item)
    return images


def format_citations(chunks: List[Dict[str, Any]], tables: Optional[List[Dict[str, Any]]] = None, images: Optional[List[Dict[str, Any]]] = None) -> str:
    pages = sorted({int(c["page"]) for c in chunks if c.get("page") is not None})
    if not pages:
        return ""
    from document_session import get_active_document
    active_doc = get_active_document() or {}
    source_name = active_doc.get("filename") or "uploaded.pdf"
    for chunk in chunks:
        if chunk.get("source") and chunk.get("source") != "uploaded.pdf":
            source_name = str(chunk["source"])
            break
    lines = [f"- [Page {p}] {source_name}" for p in pages]
    for table in tables or []:
        lines.append(f"- [Page {table.get('page')}] Table: {table.get('caption', 'PDF table')}")
    for image in images or []:
        lines.append(f"- [Page {image.get('page')}] Figure: {image.get('caption', 'PDF figure')}")
    # unique preserve order
    unique = []
    seen = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return "**Citations:**\n" + "\n".join(unique)


def ensure_citations(answer: str, chunks: List[Dict[str, Any]], tables: Optional[List[Dict[str, Any]]] = None, images: Optional[List[Dict[str, Any]]] = None) -> str:
    if _is_refusal(answer):
        return answer
    cite_block = format_citations(chunks, tables, images)
    if not cite_block:
        return answer
    body = answer
    if "**Citations:**" in body:
        body = body.split("**Citations:**")[0]
    body = re.sub(r"\nSources:\s*.*$", "", body, flags=re.IGNORECASE).rstrip()
    return body + "\n\n" + cite_block


def retrieved_pages(chunks: List[Dict[str, Any]]) -> Set[int]:
    pages = set()
    for chunk in chunks:
        page = chunk.get("page")
        if page is None:
            continue
        try:
            pages.add(int(page))
        except (TypeError, ValueError):
            continue
    return pages


def sanitize_page_citations(answer: str, chunks: List[Dict[str, Any]]) -> str:
    """Drop page markers that are not in the retrieved set."""
    allowed = retrieved_pages(chunks)
    if not answer:
        return answer

    pattern = re.compile(
        r"\[Page\s+(\d+)\]|\(Page\s+(\d+)\)|(?<![A-Za-z])Page\s+(\d+)",
        re.IGNORECASE,
    )

    def _keep(match: re.Match) -> str:
        page = int(next(g for g in match.groups() if g is not None))
        return match.group(0) if page in allowed else ""

    cleaned = pattern.sub(_keep, answer)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def citation_pages_in_answer(answer: str) -> Set[int]:
    return {int(n) for n in _PAGE_CITE.findall(answer or "")}


class RAGState(TypedDict):
    question: str
    top_k: int
    nvidia_api_key: Optional[str]
    nvidia_model: Optional[str]
    embedding_provider: Optional[str]
    pinecone_api_key: Optional[str]
    pinecone_index_name: Optional[str]
    vector_db_type: Optional[str]
    documents: List[Document]
    context_chunks: List[Dict[str, Any]]
    relevance_score: float
    is_relevant: bool
    answer: str
    grounded: bool
    confidence_score: float
    correction_notes: Optional[str]


def get_llm(
    nvidia_key: Optional[str] = None,
    nvidia_model: Optional[str] = None
):
    """
    Returns a ChatNVIDIA instance, or None for deterministic extractive synthesis.
    NVIDIA NIM (build.nvidia.com) is the only LLM provider.
    """
    nv_key = nvidia_key or config.NVIDIA_API_KEY
    if not nv_key:
        return None
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    model = nvidia_model or config.DEFAULT_NVIDIA_CHAT_MODEL
    logger.info(f"Initializing ChatNVIDIA model: {model}")
    return ChatNVIDIA(model=model, api_key=nv_key, temperature=0.0)


def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """
    Node 1: Retrieve relevant context chunks from vector store.
    Supports dynamic Pinecone credentials or loads from cached local vector store.
    """
    question = state.get("question", "").strip()
    top_k = state.get("top_k", config.TOP_K_RETRIEVAL)
    fetch_k = max(top_k * 4, 16)

    if not question:
        return {
            "documents": [],
            "context_chunks": [],
            "relevance_score": 0.0,
            "is_relevant": False
        }

    store = get_active_vector_store(
        pinecone_api_key=state.get("pinecone_api_key"),
        pinecone_index_name=state.get("pinecone_index_name"),
        nvidia_api_key=state.get("nvidia_api_key"),
        nvidia_model=state.get("nvidia_model"),
        vector_db_type=state.get("vector_db_type"),
        embedding_provider=state.get("embedding_provider")
    )

    if store is None:
        return {
            "documents": [],
            "context_chunks": [],
            "relevance_score": 0.0,
            "is_relevant": False
        }

    results = []
    if hasattr(store, "_collection"):
        raw_results = store.similarity_search_with_score(question, k=fetch_k)
        results = [(d, max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))) for d, dist in raw_results]
    elif hasattr(store, "similarity_search_with_score"):
        results = store.similarity_search_with_score(question, k=fetch_k)
    elif hasattr(store, "search_with_scores"):
        results = store.search_with_scores(question)
    else:
        docs = store.similarity_search(question, k=fetch_k)
        results = [(d, 0.70) for d in docs]

    lexical_ranked = _lexical_search_store(store, question, fetch_k)
    if lexical_ranked:
        results = _rrf_fuse(results, lexical_ranked, fetch_k)
        existing = {id(doc) for doc, _ in results}
        guaranteed = [(doc, score) for doc, score in lexical_ranked[:2] if id(doc) not in existing]
        results = guaranteed + results
    results = _expand_same_page_chunks(store, results, max_per_page=4)

    docs: List[Document] = []
    chunks: List[Dict[str, Any]] = []
    scores: List[float] = []

    for doc, score in results:
        docs.append(doc)
        normalized_score = float(score)
        scores.append(normalized_score)
        chunks.append({
            "chunk_id": doc.metadata.get("chunk_id", f"p{doc.metadata.get('page', 1)}"),
            "page": doc.metadata.get("page", 1),
            "score": round(normalized_score, 4),
            "text": doc.page_content,
            "source": doc.metadata.get("source") or "uploaded.pdf",
            "content_type": doc.metadata.get("content_type", "text"),
            "table_markdown": doc.metadata.get("table_markdown"),
            "image_path": doc.metadata.get("image_path"),
            "image_url": doc.metadata.get("image_url"),
            "caption": doc.metadata.get("caption"),
        })

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "documents": docs,
        "context_chunks": chunks,
        "relevance_score": round(avg_score, 4)
    }


def grade_relevance_node(state: RAGState) -> Dict[str, Any]:
    """
    Node 2: Keep only chunks that are both semantically similar and lexically related
    to the question. Weak matches are dropped so the generator never sees them.
    """
    question = state.get("question", "")
    top_k = state.get("top_k", config.TOP_K_RETRIEVAL)
    chunks = state.get("context_chunks", [])
    if not chunks:
        return {
            "is_relevant": False,
            "context_chunks": [],
            "relevance_score": 0.0
        }

    threshold = config.RELEVANCE_THRESHOLD
    table_intent = _is_table_question(question)
    image_intent = _is_image_question(question)
    kept: List[Dict[str, Any]] = []
    for chunk in chunks:
        lexical = _lexical_overlap(question, chunk.get("text", ""))
        ctype = chunk.get("content_type") or "text"
        if table_intent and ctype == "table" and lexical >= 0.08:
            kept.append(chunk)
            continue
        if image_intent and ctype == "image" and lexical >= 0.06:
            kept.append(chunk)
            continue
        if lexical >= 0.28:
            kept.append(chunk)
        elif chunk["score"] >= threshold and lexical >= 0.10:
            kept.append(chunk)
        elif chunk["score"] >= 0.72 and lexical >= 0.06:
            kept.append(chunk)

    kept = kept[:top_k]
    missing_entities = _missing_proper_nouns(question, kept)
    if missing_entities:
        return {
            "is_relevant": False,
            "context_chunks": [],
            "relevance_score": 0.0
        }

    # Verify that factual numbers/years in question exist in retrieved chunks
    q_nums = [
        n.replace(",", "")
        for n in re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?%?\b", question)
        if len(n.rstrip("%")) >= 3 or "%" in n
    ]
    if q_nums:
        all_context = " ".join(c.get("text", "") for c in kept)
        context_nums = set(_extract_claim_numbers(all_context))
        if any(n not in context_nums for n in q_nums):
            return {
                "is_relevant": False,
                "context_chunks": [],
                "relevance_score": 0.0
            }

    # Verify that distinctive inquiry terms have matches in kept chunks
    inquiry_terms = _distinctive_inquiry_terms(question, chunks)
    if inquiry_terms:
        all_context_l = " ".join(c.get("text", "") for c in kept).lower()
        context_tokens = set(_content_tokens(all_context_l))
        matched = {t for t in inquiry_terms if t in context_tokens or t in all_context_l}
        if not matched:
            return {
                "is_relevant": False,
                "context_chunks": [],
                "relevance_score": 0.0
            }

    if not kept:
        max_score = max(c["score"] for c in chunks)
        return {
            "is_relevant": False,
            "context_chunks": [],
            "relevance_score": round(max_score, 4)
        }

    avg_score = sum(c["score"] for c in kept) / len(kept)
    return {
        "is_relevant": True,
        "context_chunks": kept,
        "relevance_score": round(avg_score, 4)
    }


def decide_to_generate(state: RAGState) -> str:
    """
    Conditional routing edge after grading relevance.
    Routes to 'generate' if relevant chunks exist; otherwise routes to 'handle_out_of_scope'.
    """
    if state.get("is_relevant", False):
        return "generate"
    return "handle_out_of_scope"


def handle_out_of_scope_node(state: RAGState) -> Dict[str, Any]:
    """
    Out-of-scope handler when retrieved pages do not support the question.
    Eliminates hallucinations by refusing to generate on non-relevant context.
    """
    return {
        "answer": OUT_OF_SCOPE_ANSWER,
        "grounded": True,
        "confidence_score": 0.15,
        "correction_notes": "Out-of-scope query intercepted: routed away from generation to prevent hallucination."
    }


def synthesize_extractive_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Extract only sentences that share content tokens with the question.
    Never paraphrases, so it cannot introduce facts that are not in the PDF chunks.
    """
    if not chunks:
        return OUT_OF_SCOPE_ANSWER

    inquiry = _distinctive_inquiry_terms(question, chunks)
    if inquiry and not any(any(t in c.get("text", "").lower() for t in inquiry) for c in chunks):
        return OUT_OF_SCOPE_ANSWER

    q_tokens = set(_query_terms(question))
    generic = _get_corpus_frequent_terms(chunks)
    distinctive = {t for t in q_tokens - generic if len(t) >= 4}
    if not distinctive:
        distinctive = {t for t in q_tokens if len(t) >= 4}
    candidates = []
    for chunk in chunks:
        text = chunk["text"].strip()
        text_l = text.lower()
        dist_hits = [t for t in distinctive if t in text_l]
        overlap = set(_query_terms(text)) & q_tokens
        if not overlap and not dist_hits:
            continue
        if dist_hits:
            anchor = max(dist_hits, key=len)
            idx = text_l.find(anchor)
            window = text[max(0, idx - 80): idx + 420].strip()
            snippet = window
        else:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]
            snippet = max(sentences, key=len) if sentences else text[:400]
        candidates.append((
            len(dist_hits),
            1 if "%" in snippet else 0,
            len(overlap),
            len(snippet),
            chunk["page"],
            snippet
        ))

    candidates.sort(reverse=True)
    sections = []
    used_pages: Dict[int, int] = {}
    min_overlap = 1 if len(q_tokens) <= 2 else 2
    for dist, _pct, overlap, _ln, page_num, snippet in candidates:
        if dist == 0 and overlap < min_overlap:
            continue
        if used_pages.get(page_num, 0) >= 2:
            continue
        sections.append(f"**From [Page {page_num}]:**\n> {snippet}")
        used_pages[page_num] = used_pages.get(page_num, 0) + 1
        if len(sections) >= 4:
            break

    tables = collect_tables(chunks)
    if tables and (_is_table_question(question) or any(c.get("content_type") == "table" for c in chunks[:3])):
        table_parts = []
        for table in tables[:2]:
            table_parts.append(
                f"**{table.get('caption', 'PDF table')}** [Page {table.get('page')}]\n\n{table['markdown']}"
            )
        if table_parts:
            table_pages = sorted({t["page"] for t in tables[:2] if t.get("page")})
            pages_cited = table_pages or sorted({c["page"] for c in chunks[:4]})
            page_str = ", ".join(f"Page {p}" for p in pages_cited)
            return (
                f"From the uploaded PDF ({page_str}):\n\n"
                + "\n\n".join(table_parts)
            )

    if not sections:
        return OUT_OF_SCOPE_ANSWER

    pages_cited = sorted(used_pages.keys())
    page_str = ", ".join(f"Page {p}" for p in pages_cited)
    return (
        f"From the uploaded PDF ({page_str}):\n\n"
        + "\n\n".join(sections)
    )


def generate_node(state: RAGState) -> Dict[str, Any]:
    """
    Node 3: Generate grounded answer using LLM or structured extractive synthesizer.
    Strictly instructs model to cite pages and disallow external speculation.
    """
    question = state["question"]
    chunks = state.get("context_chunks", [])
    if not chunks:
        return {"answer": OUT_OF_SCOPE_ANSWER}

    tables = collect_tables(chunks)
    if tables and _is_table_question(question):
        table_parts = []
        for table in tables[:2]:
            table_parts.append(
                f"**{table.get('caption', 'PDF table')}** [Page {table.get('page')}]\n\n{table['markdown']}"
            )
        table_pages = sorted({t["page"] for t in tables[:2] if t.get("page")})
        pages_cited = table_pages or sorted({c["page"] for c in chunks})
        page_str = ", ".join(f"Page {p}" for p in pages_cited)
        return {
            "answer": (
                f"From the uploaded PDF ({page_str}):\n\n"
                + "\n\n".join(table_parts)
            )
        }

    context_lines = []
    for c in chunks:
        context_lines.append(f"--- [Page {c['page']}] ---\n{c['text']}\n")
    context_text = "\n".join(context_lines)

    system_prompt = (
        "Answer the question using only the document excerpts below.\n"
        "STRICT ANTI-HALLUCINATION RULES:\n"
        "- Cite each fact as [Page N] using a page number that appears in the excerpts.\n"
        "- Do not invent, speculate, or extrapolate facts, numbers, dates, names, or pages.\n"
        "- Never introduce outside knowledge not present in the excerpts.\n"
        "- If the excerpts do not contain sufficient information to answer the question, reply with exactly:\n"
        f"{OUT_OF_SCOPE_ANSWER}\n"
        "- If a markdown table is in the excerpts, copy it as a markdown table."
    )

    user_prompt = (
        f"Excerpts:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer from the excerpts only. Cite [Page N]."
    )

    llm = get_llm(
        nvidia_key=state.get("nvidia_api_key"),
        nvidia_model=state.get("nvidia_model")
    )

    if llm:
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = llm.invoke(messages)
            raw_answer = response.content if hasattr(response, "content") else str(response)
            if is_answer_faithful(raw_answer, chunks):
                return {"answer": raw_answer}
            logger.warning("LLM answer failed faithfulness check; using extractive fallback.")
        except Exception as e:
            logger.warning(f"LLM call failed ({e})")
            if state.get("nvidia_model") and state.get("nvidia_model") != config.DEFAULT_NVIDIA_CHAT_MODEL:
                try:
                    logger.info(f"Retrying with verified default NVIDIA model: {config.DEFAULT_NVIDIA_CHAT_MODEL}")
                    fallback_llm = get_llm(
                        nvidia_key=state.get("nvidia_api_key"),
                        nvidia_model=config.DEFAULT_NVIDIA_CHAT_MODEL
                    )
                    if fallback_llm:
                        response = fallback_llm.invoke(messages)
                        raw_answer = response.content if hasattr(response, "content") else str(response)
                        if is_answer_faithful(raw_answer, chunks):
                            return {"answer": raw_answer}
                except Exception as fb_err:
                    logger.warning(f"Fallback NVIDIA model also failed ({fb_err})")

    # High quality extractive fallback
    extractive_answer = synthesize_extractive_answer(question, chunks)
    return {"answer": extractive_answer}


def check_grounding_node(state: RAGState) -> Dict[str, Any]:
    """
    Node 4: Groundedness verification & initial confidence scoring.
    Checks that the answer's factual vocabulary and page citations correlate with retrieved context.
    """
    answer = state.get("answer", "")
    chunks = state.get("context_chunks", [])
    relevance_score = state.get("relevance_score", 0.0)

    if _is_refusal(answer) or not chunks:
        return {
            "grounded": True,
            "confidence_score": 0.15,
            "correction_notes": "Out of scope response."
        }

    context_tokens = set(_content_tokens(" ".join(c["text"] for c in chunks)))
    answer_tokens = _content_tokens(answer)
    word_overlap_ratio = (
        sum(1 for t in answer_tokens if t in context_tokens) / len(answer_tokens)
        if answer_tokens else 0.0
    )

    retrieved_pages = set(c["page"] for c in chunks)
    cited_pages = [int(p) for p in re.findall(r"\[?Page\s+(\d+)\]?", answer, re.IGNORECASE)]
    if cited_pages:
        valid_citations = [p for p in cited_pages if p in retrieved_pages]
        citation_score = len(valid_citations) / len(cited_pages)
        has_invalid_citations = (len(valid_citations) < len(cited_pages))
    else:
        citation_score = 0.0
        has_invalid_citations = False

    faithful = is_answer_faithful(answer, chunks)
    raw_confidence = (relevance_score * 0.50) + (word_overlap_ratio * 0.30) + (citation_score * 0.20)
    confidence = min(0.98, max(0.20, round(raw_confidence, 2)))
    grounded = faithful and (word_overlap_ratio >= 0.40) and not has_invalid_citations

    return {
        "grounded": grounded,
        "confidence_score": confidence,
        "correction_notes": None if grounded else "Answer contains potential ungrounded statements or citation mismatch."
    }


def decide_grounding(state: RAGState) -> str:
    """
    Conditional routing edge after checking grounding.
    Routes to 'finalize' if verified grounded; otherwise routes to 'correct_answer'.
    """
    if state.get("grounded", False):
        return "finalize"
    return "correct_answer"


def correct_answer_node(state: RAGState) -> Dict[str, Any]:
    """
    Node 5: Self-Correction node.
    Invoked when an answer fails grounding verification.
    Re-anchors the answer strictly to verified context chunks to eliminate hallucinations.
    """
    chunks = state.get("context_chunks", [])
    question = state.get("question", "")
    current_answer = state.get("answer", "")

    # Clean and re-anchor with verified extractive synthesis from top chunks
    corrected = synthesize_extractive_answer(question, chunks)
    clarification = (
        f"{corrected}\n\n"
        "*(Self-Correction Applied: replaced the draft with retrieved excerpts.)*"
    )

    return {
        "answer": clarification,
        "grounded": True,
        "confidence_score": min(state.get("confidence_score", 0.60), 0.70),
        "correction_notes": "Self-correction applied: Replaced ungrounded generation with verified context excerpts."
    }


def finalize_node(state: RAGState) -> Dict[str, Any]:
    """
    Node 6: Finalize state.
    Ensures response contract complies with requirements:
    final answer, retrieved context chunks, and confidence score.
    """
    confidence = max(0.0, min(1.0, state.get("confidence_score", 0.0)))
    return {
        "confidence_score": round(confidence, 2)
    }


def build_rag_graph():
    """
    Builds and compiles the LangGraph StateGraph pipeline with conditional routing.
    Workflow:
      START -> retrieve -> grade_relevance
      grade_relevance --(is_relevant)--> generate -> check_grounding
      grade_relevance --(not relevant)--> handle_out_of_scope -> END
      check_grounding --(grounded)--> finalize -> END
      check_grounding --(not grounded)--> correct_answer -> finalize -> END
    """
    workflow = StateGraph(RAGState)

    # Register Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_relevance", grade_relevance_node)
    workflow.add_node("handle_out_of_scope", handle_out_of_scope_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("check_grounding", check_grounding_node)
    workflow.add_node("correct_answer", correct_answer_node)
    workflow.add_node("finalize", finalize_node)

    # Register Edges & Conditional Routing
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_relevance")

    workflow.add_conditional_edges(
        "grade_relevance",
        decide_to_generate,
        {
            "generate": "generate",
            "handle_out_of_scope": "handle_out_of_scope"
        }
    )

    workflow.add_edge("generate", "check_grounding")

    workflow.add_conditional_edges(
        "check_grounding",
        decide_grounding,
        {
            "finalize": "finalize",
            "correct_answer": "correct_answer"
        }
    )

    workflow.add_edge("correct_answer", "finalize")
    workflow.add_edge("handle_out_of_scope", END)
    workflow.add_edge("finalize", END)

    return workflow.compile()


# Pre-compiled graph instance
rag_pipeline = build_rag_graph()


def query_rag(
    question: str,
    top_k: int = config.TOP_K_RETRIEVAL,
    nvidia_api_key: Optional[str] = None,
    nvidia_model: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    pinecone_api_key: Optional[str] = None,
    pinecone_index_name: Optional[str] = None,
    vector_db_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entrypoint to run question through the LangGraph RAG pipeline.
    Requires an uploaded, indexed PDF.
    """
    from document_session import require_document_ready, get_active_document
    require_document_ready()
    active = get_active_document() or {}
    vector_db_type = vector_db_type or active.get("vector_db_type")
    initial_state: RAGState = {
        "question": question,
        "top_k": top_k,
        "nvidia_api_key": nvidia_api_key,
        "nvidia_model": nvidia_model,
        "embedding_provider": embedding_provider,
        "pinecone_api_key": pinecone_api_key,
        "pinecone_index_name": pinecone_index_name,
        "vector_db_type": vector_db_type,
        "documents": [],
        "context_chunks": [],
        "relevance_score": 0.0,
        "is_relevant": False,
        "answer": "",
        "grounded": False,
        "confidence_score": 0.0,
        "correction_notes": None
    }

    final_state = rag_pipeline.invoke(initial_state)
    chunks = final_state.get("context_chunks", [])
    tables = collect_tables(chunks)
    images = collect_images(chunks)
    answer = sanitize_page_citations(final_state.get("answer", ""), chunks)
    answer = ensure_citations(answer, chunks, tables, images)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": chunks,
        "confidence_score": final_state.get("confidence_score", 0.0),
        "grounded": final_state.get("grounded", False),
        "relevance_score": final_state.get("relevance_score", 0.0),
        "correction_notes": final_state.get("correction_notes"),
        "citations": format_citations(chunks, tables, images) if not _is_refusal(answer) else "",
        "tables": tables,
        "images": images,
    }
