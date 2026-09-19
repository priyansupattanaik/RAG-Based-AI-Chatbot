# PDF Q&A (RAG Chatbot)

Upload a PDF, then ask questions. Answers come only from that file, with page citations. Chat is locked until a PDF is uploaded and indexed.

The old “ebook-only” path is gone. The bundled `Ebook-Agentic-AI.pdf` is a sample used by tests; it is **not** auto-loaded as the knowledge base.

---

## What it does

1. You upload a PDF (Streamlit sidebar or `POST /upload`).
2. PyMuPDF extracts text, markdown tables, and figures.
3. Chunks are embedded and stored (local disk cache by default; Chroma or Pinecone optional).
4. A LangGraph pipeline retrieves, grades relevance, generates, checks grounding, and self-corrects.
5. The response includes the answer, retrieved chunks, confidence, citations, tables, and images.

If nothing has been uploaded, `/chat` returns **409** and the Streamlit UI stops with an error. It will not fall back to a default PDF.

---

## Architecture

```text
[User PDF upload]  ── required before any Q&A ──
        │
        ▼
[PyMuPDF]  text + tables (markdown) + figures (PNG)
        │
        ▼
[Chunking]  RecursiveCharacterTextSplitter  (size=800, overlap=100)
        │
        ▼
[Embeddings]  NVIDIA NIM  or  local sentence-transformers (all-MiniLM-L6-v2)
        │
        ▼
[Vector store]  local (default)  |  Chroma  |  Pinecone
        │
        ▼
┌────────────────── LangGraph ──────────────────┐
│  retrieve  (dense cosine + lexical / RRF)     │
│      │                                        │
│      ▼                                        │
│  grade_relevance                              │
│     / \                                       │
│    /   \ out of scope → refuse → END          │
│   ▼                                           │
│  generate  (NVIDIA NIM, or extractive fallback)│
│      │                                        │
│      ▼                                        │
│  check_grounding  (numbers, sentences, pages) │
│     / \                                       │
│    /   \ ungrounded → correct_answer          │
│   ▼                                           │
│  finalize  +  citation sanitizer              │
└───────────────────────────────────────────────┘
        │
        ├── Streamlit UI   (`streamlit run app.py`)
        └── FastAPI        (`python run_api.py`)
```

### Grounding rules

- Answers must be supported by retrieved chunks (token overlap + claim numbers).
- Page citations that were not retrieved are stripped.
- Out-of-scope questions get a fixed refusal: the PDF does not contain enough information.
- Without an NVIDIA key, generation is extractive from the retrieved text (no invented facts).

---

## Project layout

| Path | Role |
|---|---|
| `app.py` | Streamlit UI: upload, index, chat, confidence, chunks, figures |
| `api.py` | FastAPI: `/upload`, `/chat`, `/ingest`, `/document`, `/health` |
| `run_api.py` | Server runner with Windows port-conflict fallback |
| `document_session.py` | Active-PDF session; blocks chat until ingest succeeds |
| `ingestion.py` | Extract, chunk, embed, store; tables and images |
| `rag_pipeline.py` | LangGraph RAG graph + faithfulness checks |
| `config.py` | Env-driven settings (NVIDIA, vector DB, chunking) |
| `tests/` | Unit, API, upload-gate, any-PDF grounding, port runner |

Runtime data (gitignored): `.uploads/`, `.vector_cache/`, `.chroma_db/`, `.pdf_assets/`.

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```bash
NVIDIA_API_KEY=nvapi-...          # optional; from https://build.nvidia.com
NVIDIA_CHAT_MODEL=meta/llama-3.2-11b-vision-instruct
NVIDIA_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b
VECTOR_DB_TYPE=local              # local | chroma | pinecone
```

NVIDIA is the only hosted LLM / embedding provider. Without a key, embeddings and answers use the local models.

### Streamlit

```bash
streamlit run app.py
```

1. Sidebar → **Upload PDF** → **Index uploaded PDF**.
2. Ask a question. Sample prompts are generic (`What is this document about?`).
3. Expand retrieved chunks to see page, similarity, and excerpt. Figures render inline.

### FastAPI

```bash
python run_api.py
```

`run_api.py` binds `127.0.0.1:8000` when free. On Windows `[WinError 10013]` or a busy port it prints the conflicting PID and falls back to `8001`, `8080`, `8502`, …

```bash
python run_api.py --port 8001 --host 127.0.0.1
python run_api.py --no-reload
python run_api.py --port 8000 --no-fallback
```

Docs: `http://127.0.0.1:<PORT>/docs` (root `/` redirects there).

```bash
# 1. Upload (required)
curl -X POST "http://127.0.0.1:8001/upload" -F "file=@your.pdf"

# 2. Chat
curl -X POST "http://127.0.0.1:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "top_k": 4}'
```

---

## HTTP API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/` | Redirects to `/docs` |
| `GET` | `/health` | Service status, `document_ready`, indexed chunk count, NVIDIA flags |
| `GET` | `/document` | Active upload metadata, or the “upload first” detail |
| `POST` | `/upload` | Multipart PDF. Indexes immediately. Chat stays locked until this succeeds |
| `POST` | `/chat` | JSON `{ question, top_k, … }`. **409** if no PDF is ready |
| `POST` | `/ingest` | Re-index the **already uploaded** PDF. Client `pdf_path` is ignored. **409** if none |

`/chat` response includes `answer`, `retrieved_chunks`, `confidence_score`, `grounded`, `citations`, `tables`, and `images`.

Upload rules: PDF only, `%PDF` magic bytes, empty files rejected, default 50 MB cap (`MAX_UPLOAD_BYTES`), filenames sanitized into `.uploads/` (no path escape).

---

## Vector stores

**Local (default)** — in-process cosine search, persisted at `.vector_cache/local_vector_cache.pkl`. Cache key includes PDF hash, embedding provider/model, and schema version `v4-upload-required`.

**Chroma** — embedded DB in `.chroma_db/`. Set `VECTOR_DB_TYPE=chroma` or pick it in the UI.

**Pinecone** — serverless cloud. Free starter at [pinecone.io](https://www.pinecone.io). Set:

```bash
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=rag-pdf-index
PINECONE_ENVIRONMENT=us-east-1
VECTOR_DB_TYPE=pinecone
```

The ingest path creates/validates the index dimension to match the embedding model.

---

## NVIDIA NIM

Chat models (UI picker; any catalog id via “Custom…”):

- `meta/llama-3.2-11b-vision-instruct` (default)
- `nvidia/nemotron-3.5-lightning-30b-a3b`
- `meta/llama-3.3-70b-instruct`
- `nvidia/nemotron-3-super-120b-a12b`
- `google/gemma-4-31b-it`
- `openai/gpt-oss-20b`

Embeddings: `nvidia/nemotron-3-embed-1b` (2048-d) or local `all-MiniLM-L6-v2` (384-d).

---

## Tests

```bash
python -m pytest -v tests/
```

Coverage includes:

- Upload gate: no chat / ingest / retrieval without a PDF; reject non-PDF, empty, spoofed, and path-escape uploads
- Any PDF: answers from the correct page, refuse outside facts, unicode, citation subset of retrieved pages
- Ingestion: text cleaning, chunk metadata, tables as markdown, figures
- Retrieval: cosine ranking bounds, hybrid lexical fusion, Chroma
- LangGraph: grounded generate, out-of-scope, self-correction
- Faithfulness: invented numbers and slipped-in sentences fail
- FastAPI `/health`, `/chat`, `/ingest`, `/upload`
- `run_api.py` port detection and fallback (including occupied 8000)
- NVIDIA NIM chat/embed live tests (skipped or resilient when no key)

Session fixture indexes the sample ebook into the upload session so RAG tests have a corpus. Tests that assert the gate use the `without_document` fixture.

---

## Windows port 8000 (`WinError 10013`)

`python run_api.py` already detects this and switches ports. Manually:

```powershell
Get-NetTCPConnection -LocalPort 8000
Stop-Process -Id <PID> -Force
netsh interface ipv4 show excludedportrange protocol=tcp
```

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `VECTOR_DB_TYPE` | `local` | `local` / `chroma` / `pinecone` |
| `NVIDIA_API_KEY` | empty | NIM key; empty → local embeddings + extractive answers |
| `NVIDIA_CHAT_MODEL` | `meta/llama-3.2-11b-vision-instruct` | Chat model id |
| `NVIDIA_EMBEDDING_MODEL` | `nvidia/nemotron-3-embed-1b` | Hosted embedding model |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Offline embeddings |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | Splitter |
| `TOP_K_RETRIEVAL` | `4` | Chunks returned after retrieve + grade |
| `RELEVANCE_THRESHOLD` | `0.45` | Minimum score to treat a hit as in-scope |
| `MAX_UPLOAD_BYTES` | `50MB` | Upload size cap |

Streamlit file watching is off (`.streamlit/config.toml`) so Torch/Transformers import cycles do not restart the app.
