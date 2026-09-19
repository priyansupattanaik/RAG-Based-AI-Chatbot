"""Streamlit UI: upload a PDF, then ask questions grounded in that file."""

import os
import html
from pathlib import Path

# Ensure torchvision compatibility and stubbing before any downstream library imports
from ingestion import (
    _ensure_torchvision_available,
    ingest_pipeline,
    get_active_vector_store,
    set_active_vector_store,
)
_ensure_torchvision_available()

import streamlit as st
import config
from rag_pipeline import query_rag
from document_session import (
    save_uploaded_pdf,
    mark_ingested,
    is_document_ready,
    get_active_document,
    clear_active_document,
    NO_DOCUMENT_DETAIL,
)

# Page configuration
st.set_page_config(
    page_title="PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, human-crafted UI styling
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .metric-container {
        display: flex;
        gap: 20px;
        align-items: center;
        background: #f1f5f9;
        padding: 10px 16px;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
        margin-top: 10px;
        margin-bottom: 12px;
    }
    .chunk-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .page-badge {
        background-color: #e0e7ff;
        color: #3730a3;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    .score-badge-high {
        background-color: #dcfce7;
        color: #166534;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    .score-badge-med {
        background-color: #fef9c3;
        color: #854d0e;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    .score-badge-low {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Sidebar: Configuration and Vector DB Settings
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    st.markdown("### 🗄️ Vector Database")
    vdb_choice = st.selectbox(
        "Vector Store Type",
        options=[
            "⚡ Fast Local (Zero Setup, Disk-Cached)",
            "🟣 ChromaDB (Embedded Serverless DB)",
            "🌲 Pinecone (Cloud Serverless Index)"
        ],
        index=0,
        help="Select your vector storage engine. Fast Local and ChromaDB require zero external accounts and run instantly."
    )

    pinecone_key_input = ""
    pinecone_index_input = config.PINECONE_INDEX_NAME
    pinecone_env_input = config.PINECONE_ENVIRONMENT

    if "Pinecone" in vdb_choice:
        st.info("💡 **Where to get Pinecone:** Sign up for free at [pinecone.io](https://www.pinecone.io). Starter tier gives 1 free Serverless index (100k vectors) with zero credit card needed.")
        pinecone_key_input = st.text_input(
            "Pinecone API Key",
            value=config.PINECONE_API_KEY,
            type="password",
            help="Enter Pinecone API key from app.pinecone.io"
        )
        col_idx, col_env = st.columns(2)
        with col_idx:
            pinecone_index_input = st.text_input("Index Name", value=config.PINECONE_INDEX_NAME)
        with col_env:
            pinecone_env_input = st.text_input("Region", value=config.PINECONE_ENVIRONMENT)
    elif "ChromaDB" in vdb_choice:
        st.caption("🟣 Using local ChromaDB serverless storage in `.chroma_db`.")
    else:
        st.caption("⚡ Using instant disk-cached vector store in `.vector_cache`.")

    st.markdown("---")
    st.markdown("### 🧠 NVIDIA NIM")
    st.caption("NVIDIA is the only LLM provider. Get a key at [build.nvidia.com](https://build.nvidia.com).")
    nvidia_key_input = st.text_input(
        "NVIDIA API Key",
        value=config.NVIDIA_API_KEY,
        type="password",
        help="API key from build.nvidia.com"
    )
    nvidia_model_input = config.DEFAULT_NVIDIA_CHAT_MODEL
    model_selection = st.selectbox(
        "NVIDIA Chat Model",
        options=[
            "meta/llama-3.2-11b-vision-instruct",
            "nvidia/nemotron-3.5-lightning-30b-a3b",
            "meta/llama-3.3-70b-instruct",
            "meta/llama-3.1-70b-instruct",
            "mistralai/mixtral-8x7b-instruct",
            "nvidia/nemotron-3-super-120b-a12b",
            "google/gemma-4-31b-it",
            "openai/gpt-oss-20b",
            "Custom..."
        ],
        index=0
    )
    if model_selection == "Custom...":
        nvidia_model_input = st.text_input("Custom NVIDIA Model ID", value=config.DEFAULT_NVIDIA_CHAT_MODEL)
    else:
        nvidia_model_input = model_selection

    top_k_input = st.slider("Context Chunks (Top-K)", min_value=1, max_value=8, value=4)

    st.markdown("---")
    st.markdown("### 📄 Upload PDF (required)")
    st.caption("Chat is disabled until you upload a PDF. The assistant only answers from that file.")
    uploaded = st.file_uploader("Upload a PDF knowledge source", type=["pdf"], accept_multiple_files=False)

    target_vdb_type = "local"
    if "Chroma" in vdb_choice:
        target_vdb_type = "chroma"
    elif "Pinecone" in vdb_choice:
        target_vdb_type = "pinecone"

    if uploaded is not None and st.button("📥 Index uploaded PDF", use_container_width=True, type="primary"):
        with st.spinner("Saving PDF, extracting text/tables/figures, and indexing..."):
            try:
                state = save_uploaded_pdf(uploaded.name, uploaded.getvalue())
                set_active_vector_store(None)
                emb_prov = "nvidia" if nvidia_key_input else "local"
                res = ingest_pipeline(
                    pdf_path=state["path"],
                    embedding_provider=emb_prov,
                    nvidia_api_key=nvidia_key_input or None,
                    pinecone_api_key=pinecone_key_input or None,
                    index_name=pinecone_index_input or config.PINECONE_INDEX_NAME,
                    vector_db_type=target_vdb_type,
                    use_pinecone=("Pinecone" in vdb_choice),
                    force_reindex=True
                )
                mark_ingested(
                    res["total_pages"],
                    res["total_chunks"],
                    res.get("target_store", ""),
                    vector_db_type=target_vdb_type,
                )
                st.session_state.messages = []
                st.session_state.pending_query = None
                st.success(
                    f"Indexed **{state['filename']}**: {res['total_pages']} pages, "
                    f"{res['total_chunks']} chunks in {res['target_store']}."
                )
                st.rerun()
            except Exception as e:
                clear_active_document()
                set_active_vector_store(None)
                st.error(f"Upload/ingestion failed: {e}")

    active_doc = get_active_document()
    if is_document_ready() and active_doc:
        st.success(
            f"Active PDF: **{active_doc.get('filename')}** "
            f"({active_doc.get('total_pages')} pages, {active_doc.get('total_chunks')} chunks)"
        )
        if st.button("🚀 Re-index current PDF", use_container_width=True):
            with st.spinner("Re-indexing uploaded PDF..."):
                try:
                    emb_prov = "nvidia" if nvidia_key_input else "local"
                    res = ingest_pipeline(
                        pdf_path=active_doc["path"],
                        embedding_provider=emb_prov,
                        nvidia_api_key=nvidia_key_input or None,
                        pinecone_api_key=pinecone_key_input or None,
                        index_name=pinecone_index_input or config.PINECONE_INDEX_NAME,
                        vector_db_type=target_vdb_type,
                        use_pinecone=("Pinecone" in vdb_choice),
                        force_reindex=True
                    )
                    mark_ingested(res["total_pages"], res["total_chunks"], res.get("target_store", ""))
                    st.success(f"Re-indexed {res['total_chunks']} chunks into {res['target_store']}!")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
        if st.button("🗑️ Remove PDF and lock chat", use_container_width=True):
            clear_active_document()
            set_active_vector_store(None)
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()
    else:
        st.warning(NO_DOCUMENT_DETAIL)

    st.markdown("---")
    st.caption("Answers are taken from retrieved PDF pages only. Each answer lists those page numbers.")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

# Main Header
st.title("PDF Q&A")
st.caption("Upload a PDF. Questions are answered only from that file, with page citations.")

if not is_document_ready():
    st.error(NO_DOCUMENT_DETAIL)
    st.info("Use **Upload PDF (required)** in the sidebar, then click **Index uploaded PDF**.")
    st.stop()

active_meta = get_active_document() or {}
st.caption(f"Grounded on **{active_meta.get('filename', 'uploaded.pdf')}**.")

# Example Prompt Quick Buttons
st.markdown("**Try:**")
quick_cols = st.columns(2)
quick_prompts = [
    "What is this document about? Cite pages.",
    "List the main points with page numbers.",
]
for i, col in enumerate(quick_cols):
    if col.button(quick_prompts[i], key=f"quick_{i}", use_container_width=True):
        st.session_state.pending_query = quick_prompts[i]

# Helper to render an assistant message
def render_assistant_message(msg):
    st.markdown(msg["content"])
    for img in msg.get("images") or []:
        path = img.get("path")
        if path and os.path.exists(path):
            st.image(
                path,
                caption=f"[Page {img.get('page', '?')}] {img.get('caption') or 'PDF figure'}"
            )
    score = msg.get("confidence_score", 0.0)
    grounded = msg.get("grounded", True)
    correction_notes = msg.get("correction_notes")
    pct = int(score * 100)

    if pct >= 75:
        badge_class = "score-badge-high"
    elif pct >= 45:
        badge_class = "score-badge-med"
    else:
        badge_class = "score-badge-low"

    grounded_status = "✅ Verified Grounded in PDF" if grounded else "⚠️ Flagged by Self-Correction"
    if correction_notes and "Self-correction applied" in correction_notes:
        grounded_status = "🔄 Self-Correction Applied"

    st.markdown(
        f'<div class="metric-container">'
        f'<div><strong>Confidence Score:</strong> <span class="{badge_class}">{pct}%</span></div>'
        f'<div><strong>Status:</strong> {grounded_status}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    chunks = msg.get("retrieved_chunks", [])
    if chunks:
        with st.expander(f"📚 View Retrieved Context Chunks ({len(chunks)} chunks)"):
            for idx, c in enumerate(chunks, 1):
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<div style="display:flex; justify-content:space-between; margin-bottom:8px;">'
                    f'<span class="page-badge">Page {c.get("page", "?")}</span>'
                    f'<span style="font-size:0.85rem; color:#64748b;">Similarity: <strong>{c.get("score", 0.0)}</strong> | ID: <code>{c.get("chunk_id", "")}</code></span>'
                    f'</div>'
                    f'<div style="font-size:0.92rem; color:#1e293b; white-space:pre-wrap;">{html.escape(str(c.get("text", "")))}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_message(msg)
        else:
            st.markdown(msg["content"])

# Process input
chat_input = st.chat_input("Ask a question about the uploaded PDF...")
active_query = chat_input or st.session_state.pending_query

if active_query:
    st.session_state.pending_query = None

    # Render User Query
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    emb_prov = "nvidia" if nvidia_key_input else "local"

    # Query LangGraph Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from Vector Store & running LangGraph pipeline..."):
            try:
                response = query_rag(
                    question=active_query,
                    top_k=top_k_input,
                    nvidia_api_key=nvidia_key_input or None,
                    nvidia_model=nvidia_model_input,
                    embedding_provider=emb_prov,
                    pinecone_api_key=pinecone_key_input or None,
                    pinecone_index_name=pinecone_index_input or None,
                    vector_db_type=target_vdb_type
                )

                answer_text = response.get("answer", "")
                conf_score = response.get("confidence_score", 0.0)
                is_grounded = response.get("grounded", False)
                chunks = response.get("retrieved_chunks", [])
                correction_notes = response.get("correction_notes")

                asst_msg = {
                    "role": "assistant",
                    "content": answer_text,
                    "confidence_score": conf_score,
                    "grounded": is_grounded,
                    "retrieved_chunks": chunks,
                    "correction_notes": correction_notes,
                    "images": response.get("images") or [],
                    "tables": response.get("tables") or [],
                    "citations": response.get("citations") or "",
                }

                render_assistant_message(asst_msg)
                st.session_state.messages.append(asst_msg)

            except Exception as err:
                st.error(f"Error generating answer: {err}")
