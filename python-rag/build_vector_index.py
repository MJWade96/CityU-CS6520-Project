"""Build the FAISS-backed index used by the primary native RAG pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

from llama_index.core import Document as LlamaDocument
from tqdm import tqdm

from app.rag.data.corpus_loader import build_corpus_metadata, load_corpus_chunks
from app.rag.data.data_paths import (
    COMBINED_CORPUS_FILE,
    FAISS_INDEX_DIR,
    ensure_data_directories,
)
from app.rag.data.json_utils import save_json_atomic
from app.rag.retriever.runtime_config import resolve_embedding_runtime
from app.rag.retriever.vector_store import MedicalVectorStore


CORPUS_FILE = COMBINED_CORPUS_FILE
OUTPUT_DIR = FAISS_INDEX_DIR
SKIP_TEST = False


def load_documents(corpus_file: Path) -> List[LlamaDocument]:
    """Load the combined corpus and convert it into native documents."""
    chunks = load_corpus_chunks(corpus_file)

    documents: List[LlamaDocument] = []
    for chunk in tqdm(chunks, desc="Loading corpus", unit="doc"):
        documents.append(
            LlamaDocument(
                text=chunk["content"],
                metadata=build_corpus_metadata(chunk),
            )
        )
    return documents


def build_index(
    documents: List[LlamaDocument],
    output_dir: Path,
    embedding_model_name: str,
    embedding_device: str,
) -> Dict[str, object]:
    """Embed documents and persist the native FAISS-backed index."""
    vectorstore = MedicalVectorStore(
        embedding_model_name=embedding_model_name,
        embedding_device=embedding_device,
        normalize_embeddings=True,
    )

    start_time = time.time()
    vectorstore.build(documents)
    elapsed = time.time() - start_time
    vectorstore.save(str(output_dir))

    source_counts: Dict[str, int] = {}
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    metadata = {
        "document_count": len(documents),
        "embedding_model": embedding_model_name,
        "embedding_device": embedding_device,
        "store_type": "native-faiss",
        "sources": source_counts,
        "build_time_seconds": elapsed,
    }
    save_json_atomic(output_dir / "build_metadata.json", metadata)
    return metadata


def test_retrieval(
    index_dir: Path,
    embedding_model_name: str,
    embedding_device: str,
    k: int = 5,
) -> None:
    """Run a small smoke test against the persisted index."""
    vectorstore = MedicalVectorStore(
        embedding_model_name=embedding_model_name,
        embedding_device=embedding_device,
        normalize_embeddings=True,
    )
    vectorstore.load(str(index_dir))

    for query in (
        "hypertension treatment",
        "diabetes diagnosis",
        "pneumonia antibiotics",
    ):
        print(f"\nQuery: {query}")
        for rank, node_with_score in enumerate(vectorstore.retrieve(query, k=k), start=1):
            node = node_with_score.node
            print(
                f"{rank}. [{node.metadata.get('source', 'unknown')}] {node.metadata.get('title', '')[:60]}"
            )
            print(f"   score={float(node_with_score.score or 0.0):.4f}")


def main() -> None:
    ensure_data_directories()
    documents = load_documents(Path(CORPUS_FILE))
    embedding_runtime = resolve_embedding_runtime(default_model="BAAI/bge-m3")
    metadata = build_index(
        documents,
        Path(OUTPUT_DIR),
        embedding_model_name=embedding_runtime["model_name"],
        embedding_device=embedding_runtime["device"],
    )

    print("=" * 60)
    print("Vector Index Build Complete")
    print("=" * 60)
    print(f"Documents indexed: {metadata['document_count']:,}")
    print(f"Embedding model: {metadata['embedding_model']}")
    print(f"Embedding device: {metadata['embedding_device']}")
    print(f"Sources: {metadata['sources']}")
    print(f"Build time: {metadata['build_time_seconds']:.1f}s")
    print(f"Index location: {Path(OUTPUT_DIR).resolve()}")

    if not SKIP_TEST:
        test_retrieval(
            Path(OUTPUT_DIR),
            embedding_model_name=metadata["embedding_model"],
            embedding_device=metadata["embedding_device"],
        )


if __name__ == "__main__":
    main()
