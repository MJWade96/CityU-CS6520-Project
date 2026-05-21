"""Build the native LlamaIndex FAISS index for the medical corpus."""

from __future__ import annotations

import time
from math import ceil
from typing import Dict, List

from llama_index.core import Document
from tqdm import tqdm

from app.rag.data.corpus_loader import build_corpus_metadata, load_corpus_chunks
from app.rag.data.data_paths import COMBINED_CORPUS_FILE, FAISS_INDEX_DIR
from app.rag.data.json_utils import load_json_safe, save_json_atomic
from app.rag.retriever.runtime_config import (
    DEFAULT_HF_EMBEDDING_MODEL,
    resolve_torch_device,
)
from app.rag.retriever.vector_store import MedicalVectorStore


CORPUS_FILE = COMBINED_CORPUS_FILE
INDEX_DIR = FAISS_INDEX_DIR
EMBEDDING_MODEL = DEFAULT_HF_EMBEDDING_MODEL
EMBEDDING_DEVICE = "auto"
BATCH_SIZE = 1024
INSERT_BATCH_SIZE = 8192
LOCAL_FILES_ONLY = True
USE_GPU_FAISS = True
CHECKPOINT_FILE = INDEX_DIR / "build_checkpoint.json"
BUILD_METADATA_FILE = INDEX_DIR / "build_metadata.json"


def build_documents() -> List[Document]:
    """Convert shared corpus chunks to LlamaIndex documents without duplicating mapping logic."""
    documents: List[Document] = []
    for chunk in tqdm(
        load_corpus_chunks(CORPUS_FILE),
        desc="Loading corpus",
        unit="doc",
    ):
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


def count_sources(documents: List[Document]) -> Dict[str, int]:
    """Count document sources once for the final build summary."""
    source_counts: Dict[str, int] = {}
    for doc in documents:
        source = str(doc.metadata.get("source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
    return source_counts


def corpus_fingerprint() -> Dict[str, object]:
    """Capture stable corpus inputs so resume never appends to the wrong index."""
    stat = CORPUS_FILE.stat()
    return {
        "corpus_file": str(CORPUS_FILE.resolve()),
        "corpus_size_bytes": stat.st_size,
        "corpus_mtime_ns": stat.st_mtime_ns,
    }


def checkpoint_payload(
    *,
    completed_documents: int,
    total_documents: int,
    device: str,
    elapsed: float,
) -> Dict[str, object]:
    """Build the checkpoint shape in one place to keep resume validation aligned."""
    return {
        **corpus_fingerprint(),
        "completed_documents": completed_documents,
        "total_documents": total_documents,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_device": device,
        "embedding_batch_size": BATCH_SIZE,
        "index_insert_batch_size": INSERT_BATCH_SIZE,
        "embedding_local_files_only": LOCAL_FILES_ONLY,
        "use_gpu_faiss": USE_GPU_FAISS,
        "elapsed_seconds": elapsed,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def load_resume_checkpoint(total_documents: int, device: str) -> Dict[str, object] | None:
    """Return a compatible checkpoint or fail before risking duplicate index rows."""
    if not CHECKPOINT_FILE.exists():
        return None
    if not (INDEX_DIR / "metadata.json").exists():
        raise RuntimeError(
            f"Found {CHECKPOINT_FILE}, but no persisted index metadata in {INDEX_DIR}"
        )

    checkpoint = load_json_safe(CHECKPOINT_FILE)
    expected = {
        **corpus_fingerprint(),
        "total_documents": total_documents,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_device": device,
        "embedding_batch_size": BATCH_SIZE,
        "index_insert_batch_size": INSERT_BATCH_SIZE,
        "embedding_local_files_only": LOCAL_FILES_ONLY,
        "use_gpu_faiss": USE_GPU_FAISS,
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if checkpoint.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "Index build checkpoint is incompatible with the current build "
            f"({', '.join(mismatches)} differ). Remove {CHECKPOINT_FILE} and rebuild."
        )

    completed_documents = int(checkpoint.get("completed_documents", 0))
    if completed_documents < 0 or completed_documents > total_documents:
        raise RuntimeError(
            f"Invalid completed_documents in {CHECKPOINT_FILE}: {completed_documents}"
        )
    return checkpoint


def save_build_metadata(
    *,
    documents: List[Document],
    device: str,
    elapsed: float,
) -> Dict[str, object]:
    """Persist the same summary that is printed at the end of a successful build."""
    metadata = {
        "document_count": len(documents),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_device": device,
        "embedding_batch_size": BATCH_SIZE,
        "index_insert_batch_size": INSERT_BATCH_SIZE,
        "embedding_local_files_only": LOCAL_FILES_ONLY,
        "use_gpu_faiss": USE_GPU_FAISS,
        "store_type": "native-faiss",
        "sources": count_sources(documents),
        "build_time_seconds": elapsed,
    }
    save_json_atomic(BUILD_METADATA_FILE, metadata)
    return metadata


def main() -> None:
    documents = build_documents()
    device = resolve_torch_device(EMBEDDING_DEVICE)

    print("=" * 60)
    print("Building native LlamaIndex FAISS index")
    print("=" * 60)
    print(f"Corpus: {CORPUS_FILE}")
    print(f"Documents: {len(documents):,}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding device: {device}")
    print(f"Output: {INDEX_DIR}")
    print(f"Embedding batch size: {BATCH_SIZE}")
    print(f"Embedding local files only: {LOCAL_FILES_ONLY}")
    print(f"GPU FAISS: {USE_GPU_FAISS}")
    print(f"Index insert batch size: {INSERT_BATCH_SIZE}", flush=True)

    start_time = time.time()
    resume_checkpoint = load_resume_checkpoint(len(documents), device)
    vector_store = MedicalVectorStore(
        embedding_model_name=EMBEDDING_MODEL,
        embedding_device=device,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        local_files_only=LOCAL_FILES_ONLY,
        use_gpu_faiss=USE_GPU_FAISS,
    )
    start_document = 0
    prior_elapsed = 0.0
    if resume_checkpoint:
        start_document = int(resume_checkpoint["completed_documents"])
        prior_elapsed = float(resume_checkpoint.get("elapsed_seconds", 0.0))
        next_document = min(start_document + 1, len(documents))
        print(
            "Resuming index build "
            f"from document {next_document:,}/{len(documents):,}",
            flush=True,
        )
        vector_store.load(str(INDEX_DIR))
    else:
        print("Embedding documents and building FAISS index...", flush=True)

    remaining_starts = range(start_document, len(documents), INSERT_BATCH_SIZE)
    total_remaining_batches = ceil((len(documents) - start_document) / INSERT_BATCH_SIZE)
    for batch_start in tqdm(
        remaining_starts,
        total=total_remaining_batches,
        desc="Building FAISS index",
        unit="batch",
    ):
        batch = documents[batch_start : batch_start + INSERT_BATCH_SIZE]
        vector_store.add_documents(
            batch,
            show_progress=False,
            insert_batch_size=INSERT_BATCH_SIZE,
        )
        print("Persisting index checkpoint...", flush=True)
        vector_store.save(str(INDEX_DIR))
        completed_documents = min(batch_start + len(batch), len(documents))
        elapsed = prior_elapsed + time.time() - start_time
        save_json_atomic(
            CHECKPOINT_FILE,
            checkpoint_payload(
                completed_documents=completed_documents,
                total_documents=len(documents),
                device=device,
                elapsed=elapsed,
            ),
        )
        print(
            f"[checkpoint] Indexed {completed_documents:,}/{len(documents):,} "
            f"documents in {elapsed:.1f}s",
            flush=True,
        )

    elapsed = prior_elapsed + time.time() - start_time
    metadata = save_build_metadata(documents=documents, device=device, elapsed=elapsed)
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    print("=" * 60)
    print("Vector Index Build Complete")
    print("=" * 60)
    print(f"Documents indexed: {metadata['document_count']:,}")
    print(f"Embedding model: {metadata['embedding_model']}")
    print(f"Embedding device: {metadata['embedding_device']}")
    print(f"Sources: {metadata['sources']}")
    print(f"Build time: {metadata['build_time_seconds']:.1f}s")
    print(f"Index location: {INDEX_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
