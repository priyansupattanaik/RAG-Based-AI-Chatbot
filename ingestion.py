"""
Document ingestion module for Agentic AI RAG system.
Handles PDF text extraction, text cleaning, chunking with rich metadata,
embedding generation, and storage in Pinecone (with a high-performance,
persisted local vector store fallback).
"""

import os
import re
import pickle
import hashlib
import logging
import unicodedata
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pymupdf
import numpy as np

from langchain_core.documents import Document

import config

logger = logging.getLogger("rag_ingestion")
logging.basicConfig(level=logging.INFO)

CACHE_DIR = config.BASE_DIR / ".vector_cache"
DEFAULT_CACHE_FILE = CACHE_DIR / "local_vector_cache.pkl"
PDF_ASSETS_DIR = config.PDF_ASSETS_DIR
IMAGE_ASSETS_DIR = PDF_ASSETS_DIR / "images"


def get_file_hash(file_path: str) -> str:
    """Calculates MD5 hash of a file for cache invalidation."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class LocalVectorStore:
    """
    In-memory vector store that mirrors Pinecone's interface.
    Used for local offline testing or when Pinecone API key is not configured.
    Persists embeddings to a local disk cache to avoid recomputing on every startup.
    """
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.documents: List[Document] = []
        self.embeddings_matrix: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Document]):
        if not documents:
            return
        self.documents.extend(documents)
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding_function.embed_documents(texts)
        new_matrix = np.array(vectors, dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(new_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_matrix = new_matrix / norms

        if self.embeddings_matrix is None:
            self.embeddings_matrix = new_matrix
        else:
            self.embeddings_matrix = np.vstack([self.embeddings_matrix, new_matrix])

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        if not self.documents or self.embeddings_matrix is None:
            return []

        query_vec = np.array(self.embedding_function.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        expected_dim = self.embeddings_matrix.shape[1]
        if query_vec.shape[0] != expected_dim:
            logger.warning(
                f"Dimension mismatch in LocalVectorStore: query={query_vec.shape[0]}, matrix={expected_dim}. Attempting auto-adaptation..."
            )
            cached_prov = getattr(self, "cached_provider", "local")
            cached_mod = getattr(self, "cached_model", None)
            self.embedding_function = get_embedding_model(provider=cached_prov, nvidia_model=cached_mod)
            query_vec = np.array(self.embedding_function.embed_query(query), dtype=np.float32)
            q_norm = np.linalg.norm(query_vec)
            if q_norm > 0:
                query_vec = query_vec / q_norm

        scores = np.dot(self.embeddings_matrix, query_vec)
        top_k_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_indices:
            raw_score = float(scores[idx])
            # Normalization: Cosine similarity ranges from -1.0 to 1.0.
            # In semantic document retrieval, relevance scores are bounded between 0.0 and 1.0.
            # Scores <= 0 indicate zero semantic alignment.
            normalized_score = max(0.0, min(1.0, raw_score))
            results.append((self.documents[idx], round(normalized_score, 4)))
        return results

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        k = search_kwargs.get("k", 4) if search_kwargs else 4
        class _Retriever:
            def __init__(self, store, top_k):
                self.store = store
                self.top_k = top_k
            def invoke(self, query: str) -> List[Document]:
                results = self.store.similarity_search_with_score(query, k=self.top_k)
                return [doc for doc, _ in results]
            def search_with_scores(self, query: str) -> List[Tuple[Document, float]]:
                return self.store.similarity_search_with_score(query, k=self.top_k)
        return _Retriever(self, k)

    def save_cache(
        self,
        cache_file: Path = DEFAULT_CACHE_FILE,
        file_hash: str = "",
        embedding_provider: str = "local",
        embedding_model: Optional[str] = None
    ):
        """Persists document chunks and embeddings matrix to disk with model metadata."""
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            dim = self.embeddings_matrix.shape[1] if self.embeddings_matrix is not None else 0
            payload = {
                "file_hash": file_hash,
                "schema_version": config.CACHE_SCHEMA_VERSION,
                "documents": self.documents,
                "embeddings_matrix": self.embeddings_matrix,
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "dimension": dim
            }
            with open(cache_file, "wb") as f:
                pickle.dump(payload, f)
            logger.info(f"Saved local vector cache to {cache_file} (provider={embedding_provider}, dim={dim})")
        except Exception as e:
            logger.warning(f"Failed to save local vector cache: {e}")

    def load_cache(self, cache_file: Path = DEFAULT_CACHE_FILE, expected_hash: str = "") -> bool:
        """Loads cached document chunks and embeddings matrix from disk if hash matches."""
        if not cache_file.exists():
            return False
        try:
            with open(cache_file, "rb") as f:
                payload = pickle.load(f)
            if expected_hash and payload.get("file_hash") != expected_hash:
                logger.info("Vector cache invalidated: PDF hash mismatch.")
                return False
            if payload.get("schema_version") != config.CACHE_SCHEMA_VERSION:
                logger.info("Vector cache invalidated: schema version mismatch.")
                return False
            self.documents = payload.get("documents", [])
            self.embeddings_matrix = payload.get("embeddings_matrix")
            self.cached_provider = payload.get("embedding_provider", "local")
            self.cached_model = payload.get("embedding_model")
            self.cached_dimension = payload.get("dimension")

            # Ensure embedding_function matches cached dimension if known
            if self.cached_dimension and hasattr(self, "embedding_function") and self.embedding_function is not None:
                try:
                    probe_dim = len(self.embedding_function.embed_query("probe"))
                    if probe_dim != self.cached_dimension:
                        logger.info(
                            f"Adapting LocalVectorStore embedding function from {probe_dim} to cached {self.cached_dimension} ({self.cached_provider})"
                        )
                        self.embedding_function = get_embedding_model(
                            provider=self.cached_provider,
                            nvidia_model=self.cached_model
                        )
                except Exception as ex:
                    logger.warning(f"Could not adapt embedding function dimension: {ex}")

            logger.info(f"Successfully loaded {len(self.documents)} chunks from vector cache ({cache_file})")
            return True
        except Exception as e:
            logger.warning(f"Failed to load vector cache: {e}")
            return False


# Global in-memory cache for active vector store
_GLOBAL_VECTOR_STORE = None


def _ensure_torchvision_available():
    """
    Ensures torchvision or a safe stub is present to prevent ModuleNotFoundError
    or attribute/typing inspection errors when transformers modules are inspected
    by Streamlit or other file/module watchers.
    """
    try:
        import torchvision  # noqa: F401
        import torchvision.transforms.v2  # noqa: F401
    except (ImportError, AttributeError, ModuleNotFoundError):
        import sys
        import typing
        import importlib.abc
        import importlib.machinery
        from types import ModuleType

        class _StubMeta(type):
            def __getattr__(cls, name):
                return _make_stub_class(f"{cls.__name__}.{name}")

            def __getitem__(cls, item):
                return cls

            def __or__(cls, other):
                return typing.Union[cls, other]

            def __ror__(cls, other):
                return typing.Union[other, cls]

            def __call__(cls, *args, **kwargs):
                return cls

        def _make_stub_class(name: str):
            short_name = name.split(".")[-1]
            return _StubMeta(
                short_name,
                (object,),
                {
                    "__module__": name.rsplit(".", 1)[0] if "." in name else name,
                    "__qualname__": short_name,
                    "__or__": lambda self, other: typing.Union[self.__class__, other],
                    "__ror__": lambda self, other: typing.Union[other, self.__class__],
                }
            )

        class _DynamicStubModule(ModuleType):
            def __init__(self, name: str):
                super().__init__(name)
                self.__path__ = []
                self.__file__ = f"<stub {name}>"
                self.__all__ = []

            def __getattr__(self, name: str):
                sub_name = f"{self.__name__}.{name}"
                stub = _make_stub_class(sub_name)
                setattr(self, name, stub)
                return stub

            def __or__(self, other):
                return typing.Union[typing.Any, other]

            def __ror__(self, other):
                return typing.Union[other, typing.Any]

        class _TorchVisionLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return _DynamicStubModule(spec.name)

            def exec_module(self, module):
                module.functional = module
                return None

        class _TorchVisionFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "torchvision" or fullname.startswith("torchvision."):
                    return importlib.machinery.ModuleSpec(
                        fullname, _TorchVisionLoader(), is_package=True
                    )
                return None

        if not any(isinstance(f, _TorchVisionFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _TorchVisionFinder())


# Auto-install stub hook on ingestion module load to protect all consumers
_ensure_torchvision_available()



def get_embedding_model(
    provider: str = "nvidia",
    nvidia_api_key: Optional[str] = None,
    nvidia_model: Optional[str] = None
):
    """
    Initialize embedding model.
    - 'nvidia': NVIDIA NIM embedding model (e.g. nvidia/nemotron-3-embed-1b)
    - 'local': HuggingFace sentence-transformers (all-MiniLM-L6-v2), offline fallback
    """
    nv_key = nvidia_api_key or config.NVIDIA_API_KEY
    if provider == "nvidia" and nv_key:
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        model_name = nvidia_model if (nvidia_model and "embed" in nvidia_model.lower()) else config.DEFAULT_NVIDIA_EMBEDDING_MODEL
        logger.info(f"Initializing NVIDIA NIM embeddings ({model_name})")
        return NVIDIAEmbeddings(model=model_name, api_key=nv_key)

    logger.info("Initializing local HuggingFace embeddings (all-MiniLM-L6-v2)")
    _ensure_torchvision_available()
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=config.LOCAL_EMBEDDING_MODEL)


def clean_text(raw_text: str) -> str:
    """
    Cleans raw extracted PDF text:
    - Normalizes unicode characters (NFKC)
    - Fixes broken hyphenations across line breaks
    - Removes unprintable and corrupted replacement characters
    - Normalizes whitespace
    """
    # Normalize unicode
    text = unicodedata.normalize("NFKC", raw_text)
    # Remove unmapped replacement character and common corruption artifacts
    text = text.replace("\ufffd", "").replace("\u0562", "'")
    # Fix broken hyphenations (e.g., 'transfor-\nmative' -> 'transformative')
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Normalize line breaks and tabs
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple empty lines to maximum two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    return clean_text(str(value).replace("\n", " "))


def table_to_markdown(rows: List[List[Any]]) -> str:
    """Convert a rectangular table into GitHub-flavored markdown."""
    normalized = []
    width = max((len(r) for r in rows), default=0)
    for row in rows:
        cells = [_cell_text(c) for c in row]
        while len(cells) < width:
            cells.append("")
        normalized.append(cells[:width])
    if not normalized:
        return ""
    header = [c if c else f"Col {i+1}" for i, c in enumerate(normalized[0])]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _is_real_table(rows: List[List[Any]]) -> bool:
    if not rows or len(rows) < 3 or len(rows[0]) < 2:
        return False
    header = [_cell_text(c) for c in rows[0]]
    if sum(1 for c in header if c) < 2:
        return False
    for row in rows:
        for cell in row:
            if cell and len(str(cell)) > 320:
                return False
    return True


def _page_heading(page) -> str:
    text = clean_text(page.get_text("text") or "")
    for line in text.splitlines():
        stripped = line.strip()
        if 8 <= len(stripped) <= 90 and not stripped.lower().startswith("agentic ai for executives"):
            return stripped
    return f"Page {page.number + 1} figure"


def extract_tables_and_media(pdf_path: str = str(config.DEFAULT_PDF_PATH)) -> Tuple[List[Document], List[Document], Dict[str, Any]]:
    """
    Extract markdown tables and visual figures from the PDF.
    Vector diagrams (drawings) are rendered as page figures when no large raster exists.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    IMAGE_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    source_name = os.path.basename(pdf_path)
    table_docs: List[Document] = []
    image_docs: List[Document] = []
    manifest: Dict[str, Any] = {"tables": [], "images": []}

    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            page_num = page_index + 1
            heading = _page_heading(page)
            page_text = clean_text(page.get_text("text") or "")

            try:
                found = page.find_tables()
                tables = list(found.tables) if found else []
            except Exception as exc:
                logger.warning(f"Table detection failed on page {page_num}: {exc}")
                tables = []

            for t_idx, table in enumerate(tables, 1):
                try:
                    rows = table.extract()
                except Exception:
                    continue
                if not _is_real_table(rows):
                    continue
                markdown = table_to_markdown(rows)
                if markdown.count("|") < 8:
                    continue
                caption = heading
                content = (
                    f"PDF table from page {page_num}: {caption}\n\n{markdown}"
                )
                table_docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "page": page_num,
                            "chunk_id": f"p{page_num}_table{t_idx}",
                            "source": source_name,
                            "content_type": "table",
                            "table_markdown": markdown,
                            "caption": caption,
                        }
                    )
                )
                manifest["tables"].append({
                    "page": page_num,
                    "chunk_id": f"p{page_num}_table{t_idx}",
                    "caption": caption,
                    "markdown": markdown,
                })

            saved_images = 0
            seen_xrefs = set()
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    pix = pymupdf.Pixmap(doc, xref)
                    if pix.n >= 5:
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    if pix.width < 90 or pix.height < 90:
                        continue
                    rel_name = f"images/p{page_num}_img{xref}.png"
                    out_path = PDF_ASSETS_DIR / rel_name
                    pix.save(str(out_path))
                    saved_images += 1
                    caption = heading
                    image_docs.append(
                        Document(
                            page_content=(
                                f"PDF figure on page {page_num}: {caption}. "
                                f"Nearby text: {page_text[:400]}"
                            ),
                            metadata={
                                "page": page_num,
                                "chunk_id": f"p{page_num}_img{xref}",
                                "source": source_name,
                                "content_type": "image",
                                "image_path": str(out_path),
                                "image_url": f"/assets/{rel_name.replace(chr(92), '/')}",
                                "caption": caption,
                            }
                        )
                    )
                    manifest["images"].append({
                        "page": page_num,
                        "path": str(out_path),
                        "url": f"/assets/{rel_name.replace(chr(92), '/')}",
                        "caption": caption,
                    })
                except Exception as exc:
                    logger.warning(f"Image extract failed page {page_num} xref {xref}: {exc}")

            drawings = page.get_drawings()
            if len(drawings) >= 70 and saved_images == 0:
                try:
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4), alpha=False)
                    rel_name = f"images/p{page_num}_figure.png"
                    out_path = PDF_ASSETS_DIR / rel_name
                    pix.save(str(out_path))
                    caption = heading
                    image_docs.append(
                        Document(
                            page_content=(
                                f"PDF diagram/infographic on page {page_num}: {caption}. "
                                f"Nearby text: {page_text[:400]}"
                            ),
                            metadata={
                                "page": page_num,
                                "chunk_id": f"p{page_num}_figure",
                                "source": source_name,
                                "content_type": "image",
                                "image_path": str(out_path),
                                "image_url": f"/assets/{rel_name.replace(chr(92), '/')}",
                                "caption": caption,
                            }
                        )
                    )
                    manifest["images"].append({
                        "page": page_num,
                        "path": str(out_path),
                        "url": f"/assets/{rel_name.replace(chr(92), '/')}",
                        "caption": caption,
                    })
                except Exception as exc:
                    logger.warning(f"Figure render failed page {page_num}: {exc}")

    manifest_path = PDF_ASSETS_DIR / "manifest.json"
    try:
        PDF_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception as exc:
        logger.warning(f"Could not write asset manifest: {exc}")

    logger.info(
        f"Extracted {len(table_docs)} tables and {len(image_docs)} figures from {pdf_path}"
    )
    return table_docs, image_docs, manifest


def load_asset_manifest() -> Dict[str, Any]:
    path = PDF_ASSETS_DIR / "manifest.json"
    if not path.exists():
        return {"tables": [], "images": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"tables": [], "images": []}


def extract_text_from_pdf(pdf_path: str = str(config.DEFAULT_PDF_PATH)) -> List[Document]:
    """
    Extract text page by page from the PDF file with clean metadata.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    logger.info(f"Extracting text from PDF: {pdf_path}")
    pages: List[Document] = []
    source_name = os.path.basename(pdf_path)

    with pymupdf.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            raw_text = page.get_text("text")
            cleaned = clean_text(raw_text)
            if len(re.findall(r"\w", cleaned)) >= 10:
                pages.append(
                    Document(
                        page_content=cleaned,
                        metadata={
                            "page": page_index + 1,
                            "total_pages": len(doc),
                            "source": source_name,
                            "content_type": "text",
                        }
                    )
                )

    logger.info(f"Extracted {len(pages)} clean pages from {pdf_path}")
    return pages


def chunk_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP
) -> List[Document]:
    """
    Split document pages into contextual chunks using RecursiveCharacterTextSplitter.
    Preserves page numbers, chunk sequence numbers, and filters out noise fragments.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: List[Document] = []
    chunk_counter = 0

    for doc in documents:
        page_num = doc.metadata.get("page", 1)
        sub_texts = splitter.split_text(doc.page_content)
        for i, text in enumerate(sub_texts):
            clean_chunk = text.strip()
            # Omit chunks that are too tiny to carry semantic meaning (e.g. single page headers)
            if len(clean_chunk) < 30 and len(sub_texts) > 1:
                continue
            chunk_counter += 1
            chunks.append(
                Document(
                    page_content=clean_chunk,
                    metadata={
                        "page": page_num,
                        "chunk_id": f"p{page_num}_c{i+1}",
                        "source": doc.metadata.get("source", "Ebook-Agentic-AI.pdf"),
                        "index": chunk_counter,
                        "content_type": doc.metadata.get("content_type", "text"),
                    }
                )
            )

    logger.info(f"Generated {len(chunks)} chunks from {len(documents)} pages (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def store_in_pinecone(
    chunks: List[Document],
    embeddings,
    pinecone_api_key: Optional[str] = None,
    index_name: str = config.PINECONE_INDEX_NAME,
    environment: str = config.PINECONE_ENVIRONMENT
):
    """
    Store chunks and embeddings into Pinecone vector index.
    Creates the serverless index if it doesn't already exist and validates dimensions.
    """
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore

    api_key = pinecone_api_key or config.PINECONE_API_KEY
    if not api_key:
        raise ValueError("Pinecone API key is required to store in Pinecone.")

    pc = Pinecone(api_key=api_key)
    existing_indexes = pc.list_indexes().names()

    # Determine embedding dimension
    sample_dim = len(embeddings.embed_query("dimension probe"))
    logger.info(f"Target Pinecone index: '{index_name}', vector dimension: {sample_dim}")

    if index_name not in existing_indexes:
        logger.info(f"Creating serverless Pinecone index '{index_name}' in region '{environment}'...")
        pc.create_index(
            name=index_name,
            dimension=sample_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=environment)
        )
        logger.info(f"Index '{index_name}' created successfully.")
    else:
        # Validate existing index dimension
        desc = pc.describe_index(index_name)
        if desc.dimension != sample_dim:
            raise ValueError(
                f"Existing Pinecone index '{index_name}' has dimension {desc.dimension}, "
                f"but embedding model requires {sample_dim}. Please specify a matching index or model."
            )

    # Upsert chunks via LangChain PineconeVectorStore
    logger.info(f"Upserting {len(chunks)} chunks to Pinecone index '{index_name}'...")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=api_key
    )
    logger.info("Pinecone ingestion complete!")
    return vector_store


def store_in_chroma(
    chunks: List[Document],
    embeddings,
    persist_directory: Path = config.CHROMA_PERSIST_DIR,
    collection_name: str = "agentic_ai_chunks",
    embedding_provider: str = "local",
    nvidia_model: Optional[str] = None
):
    """Store chunks and embeddings in persistent Chroma embedded vector database."""
    import chromadb
    from langchain_chroma import Chroma
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)
    sample_dim = len(embeddings.embed_query("dimension probe"))
    logger.info(f"Storing {len(chunks)} chunks in ChromaDB at {persist_directory} (dim={sample_dim})...")

    # Reset collection if exists to prevent duplication and dimension conflicts
    client = chromadb.PersistentClient(path=str(persist_directory))
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        logger.info(f"Recreating Chroma collection '{collection_name}' to ensure clean index state...")
        client.delete_collection(name=collection_name)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=collection_name
    )

    # Persist metadata for automatic retrieval alignment
    meta_path = persist_directory / "chroma_meta.json"
    meta_data = {
        "collection_name": collection_name,
        "embedding_provider": embedding_provider,
        "nvidia_model": nvidia_model or config.DEFAULT_NVIDIA_EMBEDDING_MODEL,
        "dimension": sample_dim,
        "total_chunks": len(chunks)
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not persist chroma_meta.json: {e}")

    logger.info("ChromaDB ingestion complete!")
    return vector_store


def connect_chroma(
    embeddings = None,
    persist_directory: Path = config.CHROMA_PERSIST_DIR,
    collection_name: str = "agentic_ai_chunks",
    embedding_provider: Optional[str] = None,
    nvidia_api_key: Optional[str] = None,
    nvidia_model: Optional[str] = None
):
    """Connect to existing persistent Chroma embedded database with auto dimension alignment."""
    import chromadb
    from langchain_chroma import Chroma

    persist_directory = Path(persist_directory)
    if not persist_directory.exists():
        return None

    # Load stored index metadata if present
    meta_path = persist_directory / "chroma_meta.json"
    saved_meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                saved_meta = json.load(f)
        except Exception:
            pass

    resolved_provider = embedding_provider or saved_meta.get("embedding_provider") or "local"
    resolved_model = nvidia_model or saved_meta.get("nvidia_model")

    if embeddings is None:
        embeddings = get_embedding_model(
            provider=resolved_provider,
            nvidia_api_key=nvidia_api_key,
            nvidia_model=resolved_model
        )
    elif saved_meta.get("dimension"):
        # Verify passed embedding matches collection dimension; auto-adapt if mismatched
        try:
            p_dim = len(embeddings.embed_query("probe"))
            if p_dim != saved_meta["dimension"]:
                logger.info(
                    f"Chroma collection expects dim {saved_meta['dimension']} but passed {p_dim}. "
                    f"Auto-aligning with {resolved_provider} embeddings..."
                )
                embeddings = get_embedding_model(
                    provider=resolved_provider,
                    nvidia_api_key=nvidia_api_key,
                    nvidia_model=resolved_model
                )
        except Exception as e:
            logger.warning(f"Could not probe Chroma embedding dimensions: {e}")

    try:
        client = chromadb.PersistentClient(path=str(persist_directory))
        existing_collections = [c.name for c in client.list_collections()]
        if collection_name not in existing_collections:
            return None

        store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        if store._collection.count() > 0:
            coll_meta = store._collection.metadata or {}
            dim_val = coll_meta.get("hnsw:dimension", "active")
            logger.info(f"Connected to existing ChromaDB ({store._collection.count()} chunks, dim={dim_val}).")
            return store
    except Exception as e:
        logger.warning(f"Could not connect to ChromaDB: {e}")
    return None


def connect_pinecone(
    pinecone_api_key: str,
    index_name: str = config.PINECONE_INDEX_NAME,
    embeddings = None,
    embedding_provider: Optional[str] = None,
    nvidia_api_key: Optional[str] = None,
    nvidia_model: Optional[str] = None
):
    """Connect to an existing Pinecone index without re-ingesting, auto-aligning dimensions."""
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore

    if not pinecone_api_key:
        return None

    try:
        pc = Pinecone(api_key=pinecone_api_key)
        if index_name in pc.list_indexes().names():
            desc = pc.describe_index(index_name)
            index_dim = desc.dimension

            if embeddings is None:
                if embedding_provider:
                    embeddings = get_embedding_model(
                        provider=embedding_provider,
                        nvidia_api_key=nvidia_api_key,
                        nvidia_model=nvidia_model
                    )
                elif index_dim == 2048:
                    embeddings = get_embedding_model(provider="nvidia", nvidia_api_key=nvidia_api_key, nvidia_model=nvidia_model)
                else:
                    embeddings = get_embedding_model(provider="local")
            else:
                try:
                    p_dim = len(embeddings.embed_query("probe"))
                    if p_dim != index_dim:
                        logger.info(f"Pinecone index dimension is {index_dim}, but passed embedding has {p_dim}. Auto-aligning...")
                        if index_dim == 2048:
                            embeddings = get_embedding_model(provider="nvidia", nvidia_api_key=nvidia_api_key, nvidia_model=nvidia_model)
                        else:
                            embeddings = get_embedding_model(provider="local")
                except Exception:
                    pass

            logger.info(f"Connected to existing Pinecone index '{index_name}' (dim={index_dim}).")
            return PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                pinecone_api_key=pinecone_api_key
            )
    except Exception as e:
        logger.warning(f"Could not connect to Pinecone index '{index_name}': {e}")
    return None


def ingest_pipeline(
    pdf_path: Optional[str] = None,
    embedding_provider: str = "local",
    nvidia_api_key: Optional[str] = None,
    nvidia_model: Optional[str] = None,
    pinecone_api_key: Optional[str] = None,
    index_name: str = config.PINECONE_INDEX_NAME,
    vector_db_type: Optional[str] = None,
    use_pinecone: bool = False,
    force_reindex: bool = False
) -> Dict[str, Any]:
    """
    End-to-end ingestion pipeline:
    1. Extract text from PDF with unicode normalization
    2. Chunk text with metadata
    3. Generate embeddings (NVIDIA NIM, or local sentence-transformers fallback)
    4. Store in chosen Vector DB:
       - 'pinecone': Pinecone Cloud Serverless Vector DB
       - 'chroma': Chroma Embedded Serverless Vector DB
       - 'local': High-speed disk-cached in-memory vector store
    """
    global _GLOBAL_VECTOR_STORE

    if not pdf_path:
        raise ValueError("pdf_path is required. Upload a PDF first.")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    file_hash = get_file_hash(pdf_path)
    embeddings = get_embedding_model(
        provider=embedding_provider,
        nvidia_api_key=nvidia_api_key,
        nvidia_model=nvidia_model
    )
    
    target_db = vector_db_type or config.VECTOR_DB_TYPE
    p_key = pinecone_api_key or config.PINECONE_API_KEY
    if use_pinecone and p_key:
        target_db = "pinecone"

    # Check if local cache can be reused when targeting local vector store
    if not force_reindex and target_db == "local":
        local_store = LocalVectorStore(embedding_function=embeddings)
        if local_store.load_cache(DEFAULT_CACHE_FILE, expected_hash=file_hash):
            _GLOBAL_VECTOR_STORE = local_store
            return {
                "status": "success (loaded from cache)",
                "total_pages": len(set(d.metadata.get("page") for d in local_store.documents)),
                "total_chunks": len(local_store.documents),
                "target_store": "Local Fast Vector Store (disk-cached)",
                "sample_chunk": local_store.documents[0].page_content[:200] if local_store.documents else ""
            }

    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_documents(pages)
    table_docs, image_docs, _manifest = extract_tables_and_media(pdf_path)
    for extra in table_docs + image_docs:
        chunk_counter = len(chunks) + 1
        extra.metadata["index"] = extra.metadata.get("index", chunk_counter)
        chunks.append(extra)

    store_obj = None
    target_store_desc = ""

    if target_db == "pinecone" and p_key:
        try:
            logger.info("Proceeding with Pinecone serverless ingestion...")
            store_obj = store_in_pinecone(
                chunks=chunks,
                embeddings=embeddings,
                pinecone_api_key=p_key,
                index_name=index_name
            )
            target_store_desc = f"Pinecone Serverless ({index_name})"
        except Exception as e:
            logger.warning(f"Failed to ingest into Pinecone ({e}). Falling back to local vector store.")
            local_store = LocalVectorStore(embedding_function=embeddings)
            local_store.add_documents(chunks)
            local_store.save_cache(
                DEFAULT_CACHE_FILE,
                file_hash=file_hash,
                embedding_provider=embedding_provider,
                embedding_model=nvidia_model
            )
            store_obj = local_store
            target_store_desc = f"Local Fallback ({str(e)})"
    elif target_db == "chroma":
        try:
            logger.info("Proceeding with ChromaDB embedded ingestion...")
            store_obj = store_in_chroma(
                chunks=chunks,
                embeddings=embeddings,
                embedding_provider=embedding_provider,
                nvidia_model=nvidia_model
            )
            target_store_desc = "ChromaDB Embedded (.chroma_db)"
        except Exception as e:
            logger.warning(f"Failed to ingest into Chroma ({e}). Falling back to local vector store.")
            local_store = LocalVectorStore(embedding_function=embeddings)
            local_store.add_documents(chunks)
            local_store.save_cache(
                DEFAULT_CACHE_FILE,
                file_hash=file_hash,
                embedding_provider=embedding_provider,
                embedding_model=nvidia_model
            )
            store_obj = local_store
            safe_err = str(e).encode("ascii", "ignore").decode("ascii")
            target_store_desc = f"Local Fallback ({safe_err})"
    else:
        logger.info("Using local in-memory vector store with disk cache.")
        local_store = LocalVectorStore(embedding_function=embeddings)
        local_store.add_documents(chunks)
        local_store.save_cache(
            DEFAULT_CACHE_FILE,
            file_hash=file_hash,
            embedding_provider=embedding_provider,
            embedding_model=nvidia_model
        )
        store_obj = local_store
        target_store_desc = "Local Fast Vector Store (ready)"

    _GLOBAL_VECTOR_STORE = store_obj

    return {
        "status": "success",
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "target_store": target_store_desc,
        "sample_chunk": chunks[0].page_content[:200] if chunks else ""
    }


def get_active_vector_store(
    pinecone_api_key: Optional[str] = None,
    pinecone_index_name: Optional[str] = None,
    nvidia_api_key: Optional[str] = None,
    nvidia_model: Optional[str] = None,
    vector_db_type: Optional[str] = None,
    embedding_provider: Optional[str] = None
):
    """
    Returns currently active vector store based on configuration or runtime preference:
    - 'pinecone': connects to Pinecone cloud index with auto-aligned embedding dimensions
    - 'chroma': connects to Chroma persistent DB with auto-aligned embedding dimensions
    - 'local': loads cached local vector store with auto-aligned embedding dimensions
    """
    global _GLOBAL_VECTOR_STORE

    from document_session import is_document_ready
    if not is_document_ready():
        return None

    target_db = vector_db_type or config.VECTOR_DB_TYPE
    p_key = pinecone_api_key or config.PINECONE_API_KEY
    p_idx = pinecone_index_name or config.PINECONE_INDEX_NAME

    if (target_db == "pinecone" or pinecone_api_key) and p_key:
        pine_store = connect_pinecone(
            pinecone_api_key=p_key,
            index_name=p_idx,
            embedding_provider=embedding_provider,
            nvidia_api_key=nvidia_api_key,
            nvidia_model=nvidia_model
        )
        if pine_store is not None:
            return pine_store

    if target_db == "chroma":
        chroma_store = connect_chroma(
            embedding_provider=embedding_provider,
            nvidia_api_key=nvidia_api_key,
            nvidia_model=nvidia_model
        )
        if chroma_store is not None:
            return chroma_store

    # If local vector store requested or falling back
    if target_db == "local" and DEFAULT_CACHE_FILE.exists() and not embedding_provider:
        try:
            with open(DEFAULT_CACHE_FILE, "rb") as f:
                pl = pickle.load(f)
            cached_prov = pl.get("embedding_provider")
            if cached_prov:
                embedding_provider = cached_prov
        except Exception:
            pass

    emb_prov = embedding_provider or ("nvidia" if (nvidia_api_key or config.NVIDIA_API_KEY) else "local")
    emb = get_embedding_model(
        provider=emb_prov,
        nvidia_api_key=nvidia_api_key,
        nvidia_model=nvidia_model
    )

    if _GLOBAL_VECTOR_STORE is None:
        from document_session import is_document_ready, get_active_document
        if not is_document_ready():
            logger.info("No uploaded PDF is ready; skipping auto-ingest.")
            return None
        active = get_active_document()
        logger.info("Loading vector store for uploaded PDF.")
        ingest_pipeline(
            pdf_path=active["path"],
            use_pinecone=False,
            force_reindex=False,
            embedding_provider=emb_prov,
            nvidia_api_key=nvidia_api_key,
            nvidia_model=nvidia_model,
            vector_db_type="local"
        )
    return _GLOBAL_VECTOR_STORE


def set_active_vector_store(store):
    """Overrides the active vector store instance."""
    global _GLOBAL_VECTOR_STORE
    _GLOBAL_VECTOR_STORE = store
