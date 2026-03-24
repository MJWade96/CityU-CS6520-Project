"""
Build the FAISS index used by the naive RAG pipeline.

The script reuses corpus normalization and embedding helpers so indexing logic
stays aligned with the rest of the project.
"""

from __future__ import annotations

import argparse
import json
import time
from math import ceil
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from tqdm import tqdm

from app.rag.data_paths import COMBINED_CORPUS_FILE, FAISS_INDEX_DIR, ensure_data_directories
from app.rag.embeddings import get_langchain_embeddings
from app.rag.vector_store import MedicalVectorStore


def load_documents(corpus_file: Path) -> List[Document]:
    """Load the combined corpus and convert it into LangChain documents."""
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    with corpus_file.open("r", encoding="utf-8") as handle:
        chunks = json.load(handle)

    documents: List[Document] = []
    for chunk in tqdm(chunks, desc="Loading corpus", unit="doc"):
        metadata = {
            "id": chunk.get("id", ""),
            "title": chunk.get("title", ""),
            "source": chunk.get("source", ""),
            "textbook": chunk.get("textbook", ""),
            "pmid": chunk.get("pmid", ""),
            "journal": chunk.get("journal", ""),
            "year": chunk.get("year", ""),
        }
        documents.append(Document(page_content=chunk["content"], metadata=metadata))
    return documents


def build_index(
    documents: List[Document],
    output_dir: Path,
    batch_size: int = 1000,
) -> Dict[str, object]:
    """Embed documents and persist a FAISS index."""
    embeddings = get_langchain_embeddings(
        model_type="huggingface",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = MedicalVectorStore(
        embedding_model=embeddings,
        store_type="faiss",
        persist_directory=str(output_dir),
    )

    start_time = time.time()
    total_batches = ceil(len(documents) / batch_size) if documents else 0
    batch_iterator = range(0, len(documents), batch_size)
    for start in tqdm(
        batch_iterator,
        total=total_batches,
        desc="Building FAISS index",
        unit="batch",
    ):
        batch = documents[start : start + batch_size]
        vectorstore.add_documents(batch)

    elapsed = time.time() - start_time
    vectorstore.save(str(output_dir))

    source_counts: Dict[str, int] = {}
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    metadata = {
        "document_count": len(documents),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "store_type": "faiss",
        "sources": source_counts,
        "build_time_seconds": elapsed,
    }

    with (output_dir / "build_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    return metadata


def test_retrieval(index_dir: Path, k: int = 5) -> None:
    """Run a small smoke test against the persisted index."""
    embeddings = get_langchain_embeddings(
        model_type="huggingface",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = MedicalVectorStore(
        embedding_model=embeddings,
        store_type="faiss",
        persist_directory=str(index_dir),
    )
    vectorstore.load(str(index_dir))

    for query in ("hypertension treatment", "diabetes diagnosis", "pneumonia antibiotics"):
        print(f"\nQuery: {query}")
        for rank, (doc, score) in enumerate(vectorstore.similarity_search_with_score(query, k=k), start=1):
            print(f"{rank}. [{doc.metadata.get('source', 'unknown')}] {doc.metadata.get('title', '')[:60]}")
            print(f"   score={score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build vector index for Medical RAG")
    parser.add_argument("--corpus", type=Path, default=COMBINED_CORPUS_FILE)
    parser.add_argument("--output", type=Path, default=FAISS_INDEX_DIR)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--no-test", action="store_true", help="Skip retrieval smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data_directories()
    documents = load_documents(Path(args.corpus))
    metadata = build_index(documents, Path(args.output), batch_size=args.batch_size)

    print("=" * 60)
    print("Vector Index Build Complete")
    print("=" * 60)
    print(f"Documents indexed: {metadata['document_count']:,}")
    print(f"Sources: {metadata['sources']}")
    print(f"Build time: {metadata['build_time_seconds']:.1f}s")
    print(f"Index location: {Path(args.output).resolve()}")

    if not args.no_test:
        test_retrieval(Path(args.output))


if __name__ == "__main__":
    main()
