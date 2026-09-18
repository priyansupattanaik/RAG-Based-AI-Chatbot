"""
FastAPI application exposing the Agentic AI Chatbot REST API.
Endpoints:
- POST /chat   : Query the LangGraph RAG pipeline. Returns answer, context chunks, and confidence score.
- POST /ingest : Ingest or sync the PDF into Pinecone / local vector store.
- GET  /health : Service health check with system diagnostics.
"""

import sys
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response

import config
from rag_pipeline import query_rag
from ingestion import ingest_pipeline, get_active_vector_store, set_active_vector_store
from document_session import (
    save_uploaded_pdf,
    mark_ingested,
    is_document_ready,
    get_active_document,
    require_document_ready,
    DocumentNotReady,
    NO_DOCUMENT_DETAIL,
    clear_active_document,
)

app = FastAPI(
    title="Agentic AI RAG API",
    description="Knowledge-grounded Question Answering API for 'Agentic AI: An Executive's Guide' ebook using LangGraph & Pinecone.",
    version="1.1.0"
)

# Enable CORS for cross-origin frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config.PDF_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(config.PDF_ASSETS_DIR)), name="assets")


class ChatRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "What is an AI Agent?"}, description="User question grounded in PDF")
    top_k: int = Field(default=config.TOP_K_RETRIEVAL, ge=1, le=10, description="Number of context chunks to retrieve")
    nvidia_api_key: Optional[str] = Field(default=None, description="Optional NVIDIA NIM API key override")
    nvidia_model: Optional[str] = Field(default=None, description="Optional NVIDIA model name (e.g. meta/llama-3.2-11b-vision-instruct)")
    pinecone_api_key: Optional[str] = Field(default=None, description="Optional Pinecone API key override")
    pinecone_index_name: Optional[str] = Field(default=None, description="Optional Pinecone index override")
    vector_db_type: Optional[str] = Field(default=None, description="Optional vector DB: 'local', 'chroma', or 'pinecone'")
    embedding_provider: Optional[str] = Field(default=None, description="Optional embedding provider: 'nvidia' or 'local'")


class ContextChunk(BaseModel):
    chunk_id: str
    page: int
    score: float
    text: str
    source: str
    content_type: Optional[str] = "text"
    caption: Optional[str] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    table_markdown: Optional[str] = None


class RetrievedTable(BaseModel):
    page: Optional[int] = None
    caption: Optional[str] = None
    markdown: str
    chunk_id: Optional[str] = None


class RetrievedImage(BaseModel):
    page: Optional[int] = None
    caption: Optional[str] = None
    path: Optional[str] = None
    url: Optional[str] = None
    chunk_id: Optional[str] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[ContextChunk]
    confidence_score: float
    grounded: bool
    relevance_score: Optional[float] = None
    correction_notes: Optional[str] = None
    citations: Optional[str] = None
    tables: List[RetrievedTable] = Field(default_factory=list)
    images: List[RetrievedImage] = Field(default_factory=list)


class IngestRequest(BaseModel):
    pdf_path: Optional[str] = Field(default=None, description="Ignored; ingest always uses the uploaded PDF.")
    embedding_provider: Optional[str] = Field(default="local", description="'nvidia' (NIM) or 'local' (sentence-transformers)")
    nvidia_api_key: Optional[str] = None
    nvidia_model: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    index_name: Optional[str] = config.PINECONE_INDEX_NAME
    vector_db_type: Optional[str] = Field(default="local", description="'local', 'chroma', or 'pinecone'")
    use_pinecone: bool = False
    force_reindex: bool = False


class IngestResponse(BaseModel):
    status: str
    total_pages: int
    total_chunks: int
    target_store: str
    sample_chunk: str


@app.get("/", include_in_schema=False)
def root():
    """Browser entrypoint: send people to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Avoid noisy 404s from browsers requesting a favicon."""
    return Response(status_code=204)


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint confirming service, vector store, and NVIDIA NIM status."""
    active_store = get_active_vector_store() if is_document_ready() else None
    doc_count = len(getattr(active_store, "documents", []) or [])
    if doc_count == 0 and active_store is not None and hasattr(active_store, "_collection"):
        doc_count = active_store._collection.count()

    active = get_active_document()
    return {
        "status": "healthy",
        "document_ready": is_document_ready(),
        "active_pdf": (active or {}).get("filename"),
        "vector_store_initialized": bool(is_document_ready() and active_store is not None),
        "indexed_chunks": doc_count if is_document_ready() else 0,
        "nvidia_configured": bool(config.NVIDIA_API_KEY),
        "default_nvidia_model": config.DEFAULT_NVIDIA_CHAT_MODEL,
        "default_nvidia_embedding_model": config.DEFAULT_NVIDIA_EMBEDDING_MODEL,
        "vector_db_type": config.VECTOR_DB_TYPE
    }


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """
    RAG Chat endpoint:
    Retrieves relevant PDF chunks, generates grounded answer via LangGraph pipeline,
    and returns answer, context chunks, and confidence score.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not is_document_ready():
        raise HTTPException(status_code=409, detail=NO_DOCUMENT_DETAIL)

    try:
        result = query_rag(
            question=payload.question,
            top_k=payload.top_k,
            nvidia_api_key=payload.nvidia_api_key,
            nvidia_model=payload.nvidia_model,
            embedding_provider=payload.embedding_provider,
            pinecone_api_key=payload.pinecone_api_key,
            pinecone_index_name=payload.pinecone_index_name,
            vector_db_type=payload.vector_db_type
        )
        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            retrieved_chunks=[
                ContextChunk(
                    chunk_id=c.get("chunk_id", ""),
                    page=c.get("page", 1),
                    score=c.get("score", 0.0),
                    text=c.get("text", ""),
                    source=c.get("source") or (get_active_document() or {}).get("filename") or "uploaded.pdf",
                    content_type=c.get("content_type", "text"),
                    caption=c.get("caption"),
                    image_path=c.get("image_path"),
                    image_url=c.get("image_url"),
                    table_markdown=c.get("table_markdown"),
                )
                for c in result.get("retrieved_chunks", [])
            ],
            confidence_score=result.get("confidence_score", 0.0),
            grounded=result.get("grounded", False),
            relevance_score=result.get("relevance_score"),
            correction_notes=result.get("correction_notes"),
            citations=result.get("citations") or None,
            tables=[RetrievedTable(**t) for t in result.get("tables", []) if t.get("markdown")],
            images=[RetrievedImage(**img) for img in result.get("images", []) if img.get("path") or img.get("url")],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat query: {str(e)}")


class UploadResponse(BaseModel):
    status: str
    filename: str
    total_pages: int
    total_chunks: int
    target_store: str
    sample_chunk: str


@app.post("/upload", response_model=UploadResponse)
async def upload_endpoint(
    file: UploadFile = File(..., description="PDF to index. Required before /chat."),
    embedding_provider: str = Form("local"),
    nvidia_api_key: Optional[str] = Form(None),
    nvidia_model: Optional[str] = Form(None),
    vector_db_type: str = Form("local"),
):
    """
    Upload a PDF, index it, and make it the only active corpus.
    Chat is blocked until this succeeds.
    """
    raw = await file.read()
    try:
        state = save_uploaded_pdf(file.filename or "document.pdf", raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    set_active_vector_store(None)
    try:
        res = ingest_pipeline(
            pdf_path=state["path"],
            embedding_provider=embedding_provider or ("nvidia" if config.NVIDIA_API_KEY else "local"),
            nvidia_api_key=nvidia_api_key,
            nvidia_model=nvidia_model,
            vector_db_type=vector_db_type or "local",
            force_reindex=True,
        )
        mark_ingested(
            res["total_pages"],
            res["total_chunks"],
            res.get("target_store", ""),
            vector_db_type=vector_db_type or "local",
        )
    except Exception as e:
        clear_active_document()
        set_active_vector_store(None)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}") from e

    return UploadResponse(
        status=res["status"],
        filename=state["filename"],
        total_pages=res["total_pages"],
        total_chunks=res["total_chunks"],
        target_store=res["target_store"],
        sample_chunk=res.get("sample_chunk", ""),
    )


@app.get("/document")
def document_status() -> Dict[str, Any]:
    active = get_active_document()
    return {
        "ready": is_document_ready(),
        "document": active,
        "detail": None if is_document_ready() else NO_DOCUMENT_DETAIL,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(payload: Optional[IngestRequest] = None):
    """
    Re-index the currently uploaded PDF. Rejected if nothing has been uploaded.
    """
    if payload is None:
        payload = IngestRequest()
    try:
        active = require_document_ready()
    except DocumentNotReady as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        res = ingest_pipeline(
            pdf_path=active["path"],
            embedding_provider=payload.embedding_provider or ("nvidia" if config.NVIDIA_API_KEY else "local"),
            nvidia_api_key=payload.nvidia_api_key,
            nvidia_model=payload.nvidia_model,
            pinecone_api_key=payload.pinecone_api_key,
            index_name=payload.index_name or config.PINECONE_INDEX_NAME,
            vector_db_type=payload.vector_db_type or ("pinecone" if payload.use_pinecone else "local"),
            use_pinecone=payload.use_pinecone,
            force_reindex=payload.force_reindex
        )
        mark_ingested(res["total_pages"], res["total_chunks"], res.get("target_store", ""))
        return IngestResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


if __name__ == "__main__":
    from run_api import main
    main()

