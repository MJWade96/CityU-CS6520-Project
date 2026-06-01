"""Build the native LlamaIndex FAISS index for the medical corpus."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core import Document
from tqdm import tqdm

from app.rag.data.corpus_loader import build_corpus_metadata, load_corpus_chunks
from app.rag.data.data_paths import COMBINED_CORPUS_FILE, FAISS_INDEX_DIR
from app.rag.data.json_utils import load_json_safe, save_json_atomic
from app.rag.retriever.runtime_config import (
    DEFAULT_EMBEDDING_API_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    first_env_value,
)
from app.rag.retriever.vector_store import MedicalVectorStore


CORPUS_FILE = COMBINED_CORPUS_FILE
INDEX_DIR = FAISS_INDEX_DIR
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
BATCH_SIZE = 64
INSERT_BATCH_SIZE = 8192
USE_GPU_FAISS = False
EMBEDDING_API_NUM_WORKERS = 4
INDEX_USE_ASYNC = True
EMBEDDING_API_BASE_URL = first_env_value(
    "RAG_EMBEDDING_API_BASE_URL",
    default=DEFAULT_EMBEDDING_API_BASE_URL,
)
EMBEDDING_API_KEY = first_env_value("RAG_EMBEDDING_API_KEY", "SILICONFLOW_API_KEY")
EMBEDDING_API_DIMENSIONS = None
EMBEDDING_API_TIMEOUT = 120.0
EMBEDDING_API_MAX_RETRIES = 5
CHECKPOINT_FILE = INDEX_DIR / "build_checkpoint.json"
BUILD_METADATA_FILE = INDEX_DIR / "build_metadata.json"


@dataclass(frozen=True)
class IndexBuildConfig:
    """Index build settings shared by the default builder and phase 1 runs."""

    corpus_file: Path = CORPUS_FILE
    index_dir: Path = INDEX_DIR
    embedding_model: str = EMBEDDING_MODEL
    batch_size: int = BATCH_SIZE
    insert_batch_size: int = INSERT_BATCH_SIZE
    use_gpu_faiss: bool = USE_GPU_FAISS
    embedding_api_base_url: str = EMBEDDING_API_BASE_URL
    embedding_api_key: str = field(default=EMBEDDING_API_KEY, repr=False)
    embedding_api_dimensions: Optional[int] = EMBEDDING_API_DIMENSIONS
    embedding_api_timeout: float = EMBEDDING_API_TIMEOUT
    embedding_api_max_retries: int = EMBEDDING_API_MAX_RETRIES
    embedding_api_num_workers: Optional[int] = EMBEDDING_API_NUM_WORKERS
    index_use_async: bool = INDEX_USE_ASYNC
    faiss_index_type: str = "FlatIP"
    corpus_version: str = "default"

    @property
    def checkpoint_file(self) -> Path:
        return self.index_dir / "build_checkpoint.json"

    @property
    def build_metadata_file(self) -> Path:
        return self.index_dir / "build_metadata.json"


DEFAULT_INDEX_BUILD_CONFIG = IndexBuildConfig()


def build_documents(config: IndexBuildConfig = DEFAULT_INDEX_BUILD_CONFIG) -> List[Document]:
    """Convert shared corpus chunks to LlamaIndex documents without duplicating mapping logic."""
    documents: List[Document] = []
    for chunk in tqdm(
        load_corpus_chunks(config.corpus_file),
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
        raise ValueError(f"No indexable documents found in {config.corpus_file}")
    return documents


def count_sources(documents: List[Document]) -> Dict[str, int]:
    """Count document sources once for the final build summary."""
    source_counts: Dict[str, int] = {}
    for doc in documents:
        source = str(doc.metadata.get("source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
    return source_counts


def corpus_fingerprint(config: IndexBuildConfig = DEFAULT_INDEX_BUILD_CONFIG) -> Dict[str, object]:
    """Capture stable corpus inputs so resume never appends to the wrong index."""
    stat = config.corpus_file.stat()
    return {
        "corpus_file": str(config.corpus_file.resolve()),
        "corpus_size_bytes": stat.st_size,
        "corpus_mtime_ns": stat.st_mtime_ns,
        "corpus_version": config.corpus_version,
    }


def checkpoint_payload(
    *,
    completed_documents: int,
    total_documents: int,
    elapsed: float,
    config: IndexBuildConfig = DEFAULT_INDEX_BUILD_CONFIG,
) -> Dict[str, object]:
    """Build the checkpoint shape in one place to keep resume validation aligned."""
    return {
        **corpus_fingerprint(config),
        "completed_documents": completed_documents,
        "total_documents": total_documents,
        "embedding_backend": "api",
        "embedding_model": config.embedding_model,
        "embedding_batch_size": config.batch_size,
        "index_insert_batch_size": config.insert_batch_size,
        "embedding_api_base_url": config.embedding_api_base_url,
        "embedding_api_dimensions": config.embedding_api_dimensions,
        "embedding_api_timeout": config.embedding_api_timeout,
        "embedding_api_max_retries": config.embedding_api_max_retries,
        "embedding_api_num_workers": config.embedding_api_num_workers,
        "index_use_async": config.index_use_async,
        "use_gpu_faiss": config.use_gpu_faiss,
        "faiss_index_type": config.faiss_index_type,
        "elapsed_seconds": elapsed,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def load_resume_checkpoint(
    total_documents: int,
    config: IndexBuildConfig = DEFAULT_INDEX_BUILD_CONFIG,
) -> Dict[str, object] | None:
    """Return a compatible checkpoint or fail before risking duplicate index rows."""
    if not config.checkpoint_file.exists():
        return None
    if not (config.index_dir / "metadata.json").exists():
        raise RuntimeError(
            f"Found {config.checkpoint_file}, but no persisted index metadata in {config.index_dir}"
        )

    checkpoint = load_json_safe(config.checkpoint_file)
    expected = {
        **corpus_fingerprint(config),
        "total_documents": total_documents,
        "embedding_backend": "api",
        "embedding_model": config.embedding_model,
        "embedding_batch_size": config.batch_size,
        "index_insert_batch_size": config.insert_batch_size,
        "embedding_api_base_url": config.embedding_api_base_url,
        "embedding_api_dimensions": config.embedding_api_dimensions,
        "embedding_api_timeout": config.embedding_api_timeout,
        "embedding_api_max_retries": config.embedding_api_max_retries,
        "embedding_api_num_workers": config.embedding_api_num_workers,
        "index_use_async": config.index_use_async,
        "use_gpu_faiss": config.use_gpu_faiss,
        "faiss_index_type": config.faiss_index_type,
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if checkpoint.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "Index build checkpoint is incompatible with the current build "
            f"({', '.join(mismatches)} differ). Remove {config.checkpoint_file} and rebuild."
        )

    completed_documents = int(checkpoint.get("completed_documents", 0))
    if completed_documents < 0 or completed_documents > total_documents:
        raise RuntimeError(
            f"Invalid completed_documents in {config.checkpoint_file}: {completed_documents}"
        )
    return checkpoint


def save_build_metadata(
    *,
    documents: List[Document],
    elapsed: float,
    config: IndexBuildConfig = DEFAULT_INDEX_BUILD_CONFIG,
) -> Dict[str, object]:
    """Persist the same summary that is printed at the end of a successful build."""
    metadata = {
        "document_count": len(documents),
        "embedding_backend": "api",
        "embedding_model": config.embedding_model,
        "embedding_batch_size": config.batch_size,
        "index_insert_batch_size": config.insert_batch_size,
        "embedding_api_used": True,
        "embedding_api_base_url": config.embedding_api_base_url,
        "embedding_api_dimensions": config.embedding_api_dimensions,
        "embedding_api_timeout": config.embedding_api_timeout,
        "embedding_api_max_retries": config.embedding_api_max_retries,
        "embedding_api_num_workers": config.embedding_api_num_workers,
        "index_use_async": config.index_use_async,
        "use_gpu_faiss": config.use_gpu_faiss,
        "store_type": "native-faiss",
        "faiss_index_type": config.faiss_index_type,
        "corpus_file": str(config.corpus_file.resolve()),
        "corpus_version": config.corpus_version,
        "sources": count_sources(documents),
        "build_time_seconds": elapsed,
    }
    save_json_atomic(config.build_metadata_file, metadata)
    return metadata


def build_index(config: IndexBuildConfig = DEFAULT_INDEX_BUILD_CONFIG) -> Dict[str, object]:
    """Build a FAISS index with explicit metadata for reproducible experiments."""
    documents = build_documents(config)

    print("=" * 60)
    print("Building native LlamaIndex FAISS index")
    print("=" * 60)
    print(f"Corpus: {config.corpus_file}")
    print(f"Documents: {len(documents):,}")
    print("Embedding backend: api")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Embedding API base URL: {config.embedding_api_base_url}")
    print(f"Output: {config.index_dir}")
    print(f"Embedding batch size: {config.batch_size}")
    print(f"Embedding async workers: {config.embedding_api_num_workers}")
    print(f"LlamaIndex async build: {config.index_use_async}")
    print(f"GPU FAISS: {config.use_gpu_faiss}")
    print(f"FAISS index type: {config.faiss_index_type}")
    print(f"Index insert batch size: {config.insert_batch_size}", flush=True)

    start_time = time.time()
    resume_checkpoint = load_resume_checkpoint(len(documents), config)
    vector_store = MedicalVectorStore(
        embedding_model_name=config.embedding_model,
        normalize_embeddings=True,
        batch_size=config.batch_size,
        use_gpu_faiss=config.use_gpu_faiss,
        embedding_api_base_url=config.embedding_api_base_url,
        embedding_api_key=config.embedding_api_key,
        embedding_api_dimensions=config.embedding_api_dimensions,
        embedding_api_timeout=config.embedding_api_timeout,
        embedding_api_max_retries=config.embedding_api_max_retries,
        embedding_api_num_workers=config.embedding_api_num_workers,
        index_use_async=config.index_use_async,
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
        vector_store.load(str(config.index_dir))
    else:
        print("Embedding documents and building FAISS index...", flush=True)

    remaining_starts = range(start_document, len(documents), config.insert_batch_size)
    total_remaining_batches = ceil(
        (len(documents) - start_document) / config.insert_batch_size
    )
    for batch_start in tqdm(
        remaining_starts,
        total=total_remaining_batches,
        desc="Building FAISS index",
        unit="batch",
        dynamic_ncols=True,
    ):
        batch = documents[batch_start : batch_start + config.insert_batch_size]
        batch_end = min(batch_start + len(batch), len(documents))
        batch_number = (batch_start - start_document) // config.insert_batch_size + 1
        tqdm.write(
            f"[batch {batch_number}/{total_remaining_batches}] "
            f"Indexing documents {batch_start + 1:,}-{batch_end:,} "
            f"({len(batch):,} docs)"
        )
        vector_store.add_documents(
            batch,
            show_progress=True,
            insert_batch_size=config.insert_batch_size,
        )
        tqdm.write("Persisting index checkpoint...")
        vector_store.save(str(config.index_dir))
        completed_documents = batch_end
        elapsed = prior_elapsed + time.time() - start_time
        save_json_atomic(
            config.checkpoint_file,
            checkpoint_payload(
                completed_documents=completed_documents,
                total_documents=len(documents),
                elapsed=elapsed,
                config=config,
            ),
        )
        tqdm.write(
            f"[checkpoint] Indexed {completed_documents:,}/{len(documents):,} "
            f"documents in {elapsed:.1f}s"
        )

    elapsed = prior_elapsed + time.time() - start_time
    metadata = save_build_metadata(
        documents=documents,
        elapsed=elapsed,
        config=config,
    )
    if config.checkpoint_file.exists():
        config.checkpoint_file.unlink()

    print("=" * 60)
    print("Vector Index Build Complete")
    print("=" * 60)
    print(f"Documents indexed: {metadata['document_count']:,}")
    print(f"Embedding backend: {metadata['embedding_backend']}")
    print(f"Embedding model: {metadata['embedding_model']}")
    print(f"Sources: {metadata['sources']}")
    print(f"Build time: {metadata['build_time_seconds']:.1f}s")
    print(f"Index location: {config.index_dir.resolve()}", flush=True)
    return metadata


def main() -> None:
    build_index(DEFAULT_INDEX_BUILD_CONFIG)


if __name__ == "__main__":
    main()
