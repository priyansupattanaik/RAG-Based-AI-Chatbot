# Agentic AI Knowledge Assistant (RAG Pipeline)

A production-grade, knowledge-grounded Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Pinecone**, and **Streamlit**, specifically tailored to the **"Agentic AI: An Executive's Guide"** ebook.

---

## 🎯 Requirements Fulfillment

| Requirement from Specification | Implementation Details |
|---|---|
| **1. PDF Ingestion $\to$ Chunking $\to$ Embeddings $\to$ Pinecone** | In `ingestion.py`: PyMuPDF extracts text per page with unicode normalization and de-hyphenation, `RecursiveCharacterTextSplitter` chunks text preserving metadata (`page`, `chunk_id`, `index`), embeddings are generated (NVIDIA NIM or local `all-MiniLM-L6-v2`), and upserted into Pinecone serverless index (with high-performance disk-cached local vector store). |
| **2. LangGraph RAG Pipeline** | In `rag_pipeline.py`: A compiled `StateGraph` with conditional routing: `retrieve` $\to$ `grade_relevance` $\to$ (if relevant $\to$ `generate` $\to$ `check_grounding` $\to$ (if grounded $\to$ `finalize`, else $\to$ `correct_answer`), else $\to$ `handle_out_of_scope`). |
| **3. API & UI Interface** | **Streamlit UI** (`app.py`): Full-featured interactive chat interface with confidence badges, expandable chunk viewer with page badges, and sidebar configuration.<br>**FastAPI API** (`api.py`): Exposes `/chat`, `/ingest`, and `/health` endpoints. |
| **4. Structured Response** | The pipeline returns: <br>1. **Final answer** (with page citations)<br>2. **Retrieved context chunks** (page number, similarity score, chunk ID, text)<br>3. **Confidence or score** (normalized 0.0 – 1.0 confidence score) |

---

## 🏗️ System Architecture

```text
[Ebook-Agentic-AI.pdf]
         │
         ▼
[PyMuPDF Page Extraction + Unicode Normalization]
         │
         ▼
[Recursive Character Chunking] (chunk_size=800, overlap=100)
         │
         ▼
[Embeddings] (NVIDIA NIM / Sentence-Transformers fallback)
         │
         ▼
[Pinecone Vector Store / Persisted Local Cache]
         │
         ▼
┌─────────────────────── LangGraph Workflow ────────────────────────┐
│                                                                   │
│  [START] ──► [Retrieve Top-K Chunks]                             │
│                      │                                            │
│                      ▼                                            │
│              [Grade Relevance]                                    │
│             /                 \                                   │
│    (is relevant)           (out of scope)                         │
│           /                     \                                 │
│          ▼                       ▼                                │
│   [Grounded Generation]    [Handle Out-of-Scope] ──► [END]        │
│          │                                                        │
│          ▼                                                        │
│   [Check Grounding]                                               │
│    /             \                                                │
│(grounded)    (ungrounded)                                         │
│  /                 \                                              │
│ │            [Correct Answer]                                     │
│ │             (Self-Correction)                                   │
│ │                  │                                              │
│ └──► [Finalize] ◄──┘                                              │
│          │                                                        │
│          ▼                                                        │
│        [END]                                                      │
└───────────────────────────────────────────────────────────────────┘
         │
         ├──► Streamlit Web UI (`app.py`)
         └──► FastAPI REST API (`api.py`)
```

---

## 🚀 Quick Start

### 1. Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the sample environment file:

```bash
cp .env.example .env
```

Set these in `.env` (NVIDIA is the **only** LLM provider):
- `NVIDIA_API_KEY` — from [build.nvidia.com](https://build.nvidia.com)
- `NVIDIA_CHAT_MODEL=meta/llama-3.2-11b-vision-instruct`
- `NVIDIA_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b`
- `VECTOR_DB_TYPE=local` (Options: `local`, `chroma`, `pinecone`)

---

## 🌲 Vector Database Setup

### 1. Where and How to Get Pinecone (Cloud Serverless)
If you want to use cloud-hosted Pinecone:
1. **Sign Up**: Go to [https://www.pinecone.io](https://www.pinecone.io) and register for a free account.
2. **Starter Tier**: Pinecone's Starter plan is **100% free forever** with **1 Serverless project** and up to **100,000 vectors** (AWS `us-east-1`), with **no credit card required**.
3. **Get Your API Key**:
   - In the Pinecone console, go to **API Keys** in the left sidebar.
   - Copy your key and paste it into `.env`:
     ```bash
     PINECONE_API_KEY=your-pinecone-api-key-here
     PINECONE_INDEX_NAME=agentic-ai-index
     PINECONE_ENVIRONMENT=us-east-1
     ```
   - Alternatively, enter it directly into the Streamlit UI sidebar.
4. **Auto-Provision**: The chatbot will automatically validate dimensions and create the serverless index for you upon first ingest!

### 2. Fast Serverless / Embedded Vector DBs (No Account Needed)
If you do not want to create a cloud account or wait for Pinecone:
- ⚡ **Local Fast Vector Store (Default)**:
  - Built directly into `ingestion.py`.
  - Serializes normalized embeddings with sub-millisecond cosine similarity into `.vector_cache/local_vector_cache.pkl`.
  - Zero latency, zero cloud dependencies, instant startup.
- 🟣 **ChromaDB (Embedded Serverless)**:
  - Built-in persistent vector database running locally in `.chroma_db/`.
  - Simply select `ChromaDB` from the dropdown in the UI or set `VECTOR_DB_TYPE=chroma` in `.env`.

---

## 🟢 NVIDIA NIM Models (build.nvidia.com / try.nvidia.com)

The assistant natively supports NVIDIA's catalog of hosted NIM models using `langchain-nvidia-ai-endpoints`:
- **Active Chat Models**:
  - `meta/llama-3.2-11b-vision-instruct` (Default — extremely fast, responsive)
  - `nvidia/nemotron-3.5-lightning-30b-a3b`
  - `nvidia/nemotron-3-super-120b-a12b`
  - `google/gemma-4-31b-it`
  - `openai/gpt-oss-20b`
- **Active Embedding Models**:
  - `nvidia/nemotron-3-embed-1b` (2048 dimensions)
  - `nvidia/llama-nemotron-embed-vl-1b-v2` (2048 dimensions)

---

### 3. Run Streamlit UI

```bash
streamlit run app.py
```

Features in the UI:
- **Vector DB Selector**: Switch between Local Fast Cache, ChromaDB, and Pinecone Serverless with 1 click.
- **NVIDIA NIM**: Chat model picker and API key (OpenAI/Groq are not supported).
- **Visual Confidence Score**: Color-coded confidence percentage (Green/Yellow/Red).
- **Retrieved Chunks Inspector**: Expandable cards displaying source page numbers, similarity scores, and excerpts.
- **Quick Sample Questions**: Pre-configured prompts to test book-specific topics instantly.

### 4. Run FastAPI Backend

#### Option A: Robust Runner with Auto-Fallback (Recommended)
Automatically detects if port 8000 is occupied or blocked by Windows `[WinError 10013]`, identifies the conflicting PID, prints kill commands, and seamlessly switches to an available port (e.g. 8001, 8080, 8502):
```bash
python run_api.py
```
Or directly via `api.py`:
```bash
python api.py
```
Custom host, port, and behavior flags are fully supported on both scripts:
```bash
# Custom port and host
python run_api.py --port 8001 --host 127.0.0.1

# Disable reload in production or background runs
python run_api.py --no-reload

# Enforce strict port binding without automatic fallback (fails fast if occupied)
python run_api.py --port 8000 --no-fallback
```

#### Option B: Standard Uvicorn Command
If running via standard uvicorn, specify an available port:
```bash
uvicorn api:app --reload --port 8001
```

Interactive API documentation available at: `http://127.0.0.1:<PORT>/docs` (e.g., `http://127.0.0.1:8001/docs`)

#### 🛠️ Troubleshooting: Windows `[WinError 10013]` on Port 8000
If you encounter `ERROR: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions`, port 8000 is either occupied by another process (such as an existing background uvicorn server) or reserved by Windows:

1. **Automatic Detection:**
   Running `python run_api.py` automatically inspects the TCP table, prints the exact conflicting PID and process name (e.g., `PID 18408 (python.exe)`), and switches to port 8001.

2. **Identify the process manually:**
   ```cmd
   netstat -ano | findstr :8000
   ```
   Or in PowerShell:
   ```powershell
   Get-NetTCPConnection -LocalPort 8000
   ```

3. **Terminate the conflicting process (replace `<PID>` with the actual PID):**
   ```cmd
   taskkill /PID <PID> /F
   ```
   Or in PowerShell:
   ```powershell
   Stop-Process -Id <PID> -Force
   ```

4. **Check Windows reserved/excluded port ranges (Hyper-V / WSL):**
   ```cmd
   netsh interface ipv4 show excludedportrange protocol=tcp
   ```

5. **Or simply run on an alternative open port:**
   ```bash
   python run_api.py
   # or
   uvicorn api:app --reload --port 8001
   ```

#### Sample Chat Request:
```bash
curl -X POST "http://127.0.0.1:8001/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is an AI Agent according to the ebook?",
       "top_k": 2,
       "nvidia_model": "meta/llama-3.2-11b-vision-instruct"
     }'
```


---

## 🧪 Automated Testing

To run the automated test suite:

```bash
python -m pytest -v tests/test_rag.py
```

Includes 15 automated unit and integration tests:
- Unicode text cleaning and ligature normalization
- PDF page extraction and metadata preservation
- Document chunking consistency
- Cosine similarity ranking and score boundary validation
- LangGraph grounded RAG pipeline execution
- Conditional out-of-scope question routing
- Self-correction activation on ungrounded generation
- FastAPI endpoints (`/health`, `/chat`, `/ingest`)
- NVIDIA NIM Chat model live invocation
- NVIDIA NIM Embeddings dimension & vector verification
- ChromaDB embedded serverless vector storage and retrieval
- FastAPI NVIDIA end-to-end chat endpoint
