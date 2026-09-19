"""
Configuration settings for the Agentic AI RAG system.
Loads environment variables and sets defaults for chunking, models, and retrieval.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_NAME = os.getenv("DEFAULT_PDF_NAME", "Ebook-Agentic-AI.pdf")
DEFAULT_PDF_PATH = Path(os.getenv("DEFAULT_PDF_PATH", str(BASE_DIR / DEFAULT_PDF_NAME)))

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-pdf-index")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pdf_rag_chunks")

# NVIDIA NIM is the only LLM / hosted-embedding provider
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
DEFAULT_NVIDIA_CHAT_MODEL = os.getenv("NVIDIA_CHAT_MODEL", "meta/llama-3.2-11b-vision-instruct")
DEFAULT_NVIDIA_EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b")
# Local sentence-transformers fallback for offline tests / no NVIDIA key
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Vector Database configuration
# Options: "local" (instant disk-cached vector store, zero cloud setup), "chroma" (embedded serverless), "pinecone" (cloud serverless)
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "local")
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / ".chroma_db")))
PDF_ASSETS_DIR = Path(os.getenv("PDF_ASSETS_DIR", str(BASE_DIR / ".pdf_assets")))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / ".uploads")))
CACHE_SCHEMA_VERSION = os.getenv("CACHE_SCHEMA_VERSION", "v4-upload-required")

# Chunking & Retrieval settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "4"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.45"))
