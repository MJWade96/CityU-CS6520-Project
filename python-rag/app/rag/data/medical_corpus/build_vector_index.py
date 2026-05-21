"""Build the native LlamaIndex FAISS index for the medical corpus."""

from __future__ import annotations

from typing import List

from llama_index.core import Document

from app.rag.data.corpus_loader import build_corpus_metadata, load_corpus_chunks
from app.rag.data.data_paths import COMBINED_CORPUS_FILE, FAISS_INDEX_DIR
from app.rag.retriever.runtime_config import (
    DEFAULT_HF_EMBEDDING_MODEL,
    resolve_torch_device,
)
from app.rag.retriever.vector_store import MedicalVectorStore


CORPUS_FILE = COMBINED_CORPUS_FILE
INDEX_DIR = FAISS_INDEX_DIR
EMBEDDING_MODEL = DEFAULT_HF_EMBEDDING_MODEL
EMBEDDING_DEVICE = "auto"
BATCH_SIZE = 256


def build_documents() -> List[Document]:
    """Convert shared corpus chunks to LlamaIndex documents without duplicating mapping logic."""
    documents: List[Document] = []
    for chunk in load_corpus_chunks(CORPUS_FILE):
        text = str(chunk.get("content") or chunk.get("text") or "").strip()
        if not text:
            continue
        documents.append(
            Document(
                text=text,
                metadata=build_corpus_metadata(chunk),
            )
        )
    if not documents:
        raise ValueError(f"No indexable documents found in {CORPUS_FILE}")
    return documents


def main() -> None:
    documents = build_documents()
    device = resolve_torch_device(EMBEDDING_DEVICE)
    vector_store = MedicalVectorStore(
        embedding_model_name=EMBEDDING_MODEL,
        embedding_device=device,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
    )

    print("=" * 60)
    print("Building native LlamaIndex FAISS index")
    print("=" * 60)
    print(f"Corpus: {CORPUS_FILE}")
    print(f"Documents: {len(documents):,}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding device: {device}")
    print(f"Output: {INDEX_DIR}")

    vector_store.build(documents)
    vector_store.save(str(INDEX_DIR))

    print("Index build complete")


if __name__ == "__main__":
    main()
