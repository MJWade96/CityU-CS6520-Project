"""Generate reusable local BGE chunk embeddings on AutoDL.

This script mirrors the formal API index input text while changing only the
embedding generation backend. It writes artifact arrays consumed later by the
formal evaluator on the PC.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from app.rag.data.corpus_registry import combine_registered_corpora
from app.rag.data.data_paths import RESULT_INDEXES_DIR, ensure_data_directories
from app.rag.data.json_utils import load_json_safe, save_json_atomic
from app.rag.experiments.phase1_formal_ablation import (
    CORPUS_VARIANTS,
    FAISS_INDEX_TYPE,
    LOCAL_EMBEDDING_BACKENDS,
    EMBEDDING_PROVIDERS,
    _slug,
)


CORPUS_VERSIONS_TO_EMBED = ("statpearls", "statpearls_textbooks")
LOCAL_BGE_MODELS = tuple(
    provider.model
    for provider in EMBEDDING_PROVIDERS
    if provider.backend == "local_hf_embedding"
)
EMBEDDING_BACKEND = "local_hf_embedding"
CORPUS_BATCH_SIZE = 128
SOURCE_RUNTIME = "autodl"
EMBEDDING_INPUT_FORMAT = "corpus_content_text"
CHECKPOINT_FILENAME = "local_embedding_checkpoint.json"


def _artifact_dir(corpus_version: str, embedding_model: str):
    return (
        RESULT_INDEXES_DIR
        / f"{_slug(corpus_version)}__{_slug(embedding_model)}__{FAISS_INDEX_TYPE}"
    )


def _checkpoint_path(artifact_dir: Path) -> Path:
    return artifact_dir / CHECKPOINT_FILENAME


def _embedding_path(artifact_dir: Path) -> Path:
    return artifact_dir / "chunk_embeddings.npy"


def _manifest_path(artifact_dir: Path) -> Path:
    return artifact_dir / "manifest.json"


def _load_corpus_records(corpus_version: str) -> Tuple[List[Dict[str, Any]], Sequence[str]]:
    if corpus_version not in CORPUS_VARIANTS:
        known = ", ".join(sorted(CORPUS_VARIANTS))
        raise KeyError(f"Unknown corpus version {corpus_version!r}; expected one of {known}")

    selected_sources = CORPUS_VARIANTS[corpus_version]
    result = combine_registered_corpora(selected_sources=selected_sources)
    records = [
        {
            "doc_id": str(record.get("id") or f"{corpus_version}-{index}"),
            "title": str(record.get("title") or "").strip(),
            "content": str(record.get("content") or record.get("text") or "").strip(),
            "source": str(record.get("source") or "unknown"),
        }
        for index, record in enumerate(result["records"])
        if str(record.get("content") or record.get("text") or "").strip()
    ]
    if not records:
        raise ValueError(f"No embeddable corpus records found for {corpus_version}")
    return records, selected_sources


def _load_huggingface_embedding_model(model_name: str) -> Any:
    import torch
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        embed_batch_size=CORPUS_BATCH_SIZE,
    )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype("float32")


def _checkpoint_payload(
    *,
    corpus_version: str,
    selected_sources: Sequence[str],
    document_count: int,
    embedding_model: str,
    embedding_dim: int,
    completed_documents: int,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    return {
        "corpus_version": corpus_version,
        "selected_sources": list(selected_sources),
        "total_documents": document_count,
        "completed_documents": completed_documents,
        "embedding_model": embedding_model,
        "embedding_backend": EMBEDDING_BACKEND,
        "embedding_dim": embedding_dim,
        "embedding_batch_size": CORPUS_BATCH_SIZE,
        "embedding_input_format": EMBEDDING_INPUT_FORMAT,
        "source_runtime": SOURCE_RUNTIME,
        "elapsed_seconds": elapsed_seconds,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _compatible_complete_artifact(
    *,
    artifact_dir: Path,
    corpus_version: str,
    selected_sources: Sequence[str],
    document_count: int,
    embedding_model: str,
) -> bool:
    embedding_path = _embedding_path(artifact_dir)
    manifest_path = _manifest_path(artifact_dir)
    if not embedding_path.exists() or not manifest_path.exists():
        return False

    manifest = load_json_safe(manifest_path)
    expected = {
        "corpus_version": corpus_version,
        "selected_sources": list(selected_sources),
        "document_count": document_count,
        "embedding_model": embedding_model,
        "embedding_backend": EMBEDDING_BACKEND,
        "embedding_input_format": EMBEDDING_INPUT_FORMAT,
        "source_runtime": SOURCE_RUNTIME,
    }
    mismatches = [
        key for key, expected_value in expected.items() if manifest.get(key) != expected_value
    ]
    if mismatches:
        return False

    embeddings = np.load(embedding_path, mmap_mode="r")
    return (
        len(embeddings.shape) == 2
        and int(embeddings.shape[0]) == document_count
        and int(embeddings.shape[1]) == int(manifest.get("embedding_dim", -1))
    )


def _load_resume_checkpoint(
    *,
    artifact_dir: Path,
    corpus_version: str,
    selected_sources: Sequence[str],
    document_count: int,
    embedding_model: str,
) -> Dict[str, Any] | None:
    checkpoint_path = _checkpoint_path(artifact_dir)
    embedding_path = _embedding_path(artifact_dir)
    if not checkpoint_path.exists():
        return None
    if not embedding_path.exists():
        raise RuntimeError(
            f"Found {checkpoint_path}, but missing {embedding_path}. "
            "Remove the checkpoint or restore the partial embedding file."
        )

    checkpoint = load_json_safe(checkpoint_path)
    expected = {
        "corpus_version": corpus_version,
        "selected_sources": list(selected_sources),
        "total_documents": document_count,
        "embedding_model": embedding_model,
        "embedding_backend": EMBEDDING_BACKEND,
        "embedding_batch_size": CORPUS_BATCH_SIZE,
        "embedding_input_format": EMBEDDING_INPUT_FORMAT,
        "source_runtime": SOURCE_RUNTIME,
    }
    mismatches = [
        key for key, expected_value in expected.items() if checkpoint.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "Local embedding checkpoint is incompatible with the current build "
            f"({', '.join(mismatches)} differ). Remove {checkpoint_path} and rebuild."
        )

    completed_documents = int(checkpoint.get("completed_documents", 0))
    embedding_dim = int(checkpoint.get("embedding_dim", 0))
    if completed_documents < 0 or completed_documents > document_count:
        raise RuntimeError(
            f"Invalid completed_documents in {checkpoint_path}: {completed_documents}"
        )
    if embedding_dim <= 0:
        raise RuntimeError(f"Invalid embedding_dim in {checkpoint_path}: {embedding_dim}")

    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.shape != (document_count, embedding_dim):
        raise RuntimeError(
            f"Partial embedding file shape {embeddings.shape} does not match checkpoint "
            f"shape {(document_count, embedding_dim)} for {embedding_path}"
        )
    return checkpoint


def _empty_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def embed_texts(
    embed_model: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
    progress_label: str,
) -> np.ndarray:
    """Embed in explicit chunks so long corpus texts cannot form one huge GPU batch."""
    batches: List[np.ndarray] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = list(texts[start : start + batch_size])
        vectors = embed_model.get_text_embedding_batch(batch, show_progress=True)
        batches.append(np.asarray(vectors, dtype="float32"))
        completed = min(start + len(batch), total)
        print(
            f"  {progress_label} embedded {completed:,}/{total:,} texts "
            f"(batch_size={batch_size})",
            flush=True,
        )
        _empty_cuda_cache()
    return _normalize_rows(np.vstack(batches))


def embed_texts_to_checkpointed_file(
    embed_model: Any,
    texts: Sequence[str],
    *,
    artifact_dir: Path,
    corpus_version: str,
    selected_sources: Sequence[str],
    embedding_model: str,
    batch_size: int,
    progress_label: str,
) -> Tuple[int, float]:
    embedding_path = _embedding_path(artifact_dir)
    checkpoint_path = _checkpoint_path(artifact_dir)
    total = len(texts)
    started_at = time.time()
    prior_elapsed = 0.0
    completed_documents = 0
    embedding_dim: int | None = None
    output: np.memmap | None = None

    checkpoint = _load_resume_checkpoint(
        artifact_dir=artifact_dir,
        corpus_version=corpus_version,
        selected_sources=selected_sources,
        document_count=total,
        embedding_model=embedding_model,
    )
    if checkpoint:
        completed_documents = int(checkpoint["completed_documents"])
        embedding_dim = int(checkpoint["embedding_dim"])
        prior_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        output = np.lib.format.open_memmap(embedding_path, mode="r+")
        print(
            f"Resuming embedding corpus={corpus_version}, model={embedding_model} "
            f"from document {completed_documents + 1:,}/{total:,}",
            flush=True,
        )

    for start in range(completed_documents, total, batch_size):
        batch = list(texts[start : start + batch_size])
        vectors = np.asarray(
            embed_model.get_text_embedding_batch(batch, show_progress=True),
            dtype="float32",
        )
        vectors = _normalize_rows(vectors)
        if embedding_dim is None:
            embedding_dim = int(vectors.shape[1])
            output = np.lib.format.open_memmap(
                embedding_path,
                mode="w+",
                dtype="float32",
                shape=(total, embedding_dim),
            )
        if vectors.shape[1] != embedding_dim:
            raise RuntimeError(
                f"Embedding dimension changed from {embedding_dim} to {vectors.shape[1]}"
            )
        if output is None:
            raise RuntimeError("Embedding output file was not initialized")

        batch_end = min(start + len(batch), total)
        output[start:batch_end] = vectors
        output.flush()
        completed_documents = batch_end
        elapsed = prior_elapsed + time.time() - started_at
        save_json_atomic(
            checkpoint_path,
            _checkpoint_payload(
                corpus_version=corpus_version,
                selected_sources=selected_sources,
                document_count=total,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                completed_documents=completed_documents,
                elapsed_seconds=elapsed,
            ),
        )
        print(
            f"  {progress_label} embedded {completed_documents:,}/{total:,} texts "
            f"(batch_size={batch_size})",
            flush=True,
        )
        _empty_cuda_cache()

    if output is not None:
        output.flush()
    if embedding_dim is None:
        raise RuntimeError(f"No embeddings were written for {progress_label}")
    return embedding_dim, prior_elapsed + time.time() - started_at


def _write_manifest(
    *,
    corpus_version: str,
    selected_sources: Sequence[str],
    document_count: int,
    embedding_model: str,
    embedding_dim: int,
    artifact_dir,
    elapsed_seconds: float,
) -> None:
    save_json_atomic(
        artifact_dir / "manifest.json",
        {
            "corpus_version": corpus_version,
            "selected_sources": list(selected_sources),
            "document_count": document_count,
            "embedding_model": embedding_model,
            "embedding_backend": EMBEDDING_BACKEND,
            "embedding_dim": embedding_dim,
            "embedding_input_format": EMBEDDING_INPUT_FORMAT,
            "chunk_embeddings_path": str(artifact_dir / "chunk_embeddings.npy"),
            "source_runtime": SOURCE_RUNTIME,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "build_time_seconds": elapsed_seconds,
        },
    )


def embed_corpus_version(
    embed_model: Any,
    *,
    embedding_model: str,
    corpus_version: str,
) -> None:
    records, selected_sources = _load_corpus_records(corpus_version)
    artifact_dir = _artifact_dir(corpus_version, embedding_model)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    texts = [str(record["content"]) for record in records]

    if _compatible_complete_artifact(
        artifact_dir=artifact_dir,
        corpus_version=corpus_version,
        selected_sources=selected_sources,
        document_count=len(records),
        embedding_model=embedding_model,
    ):
        print(
            f"Skipping completed local embedding artifact corpus={corpus_version}, "
            f"model={embedding_model}, output={artifact_dir}",
            flush=True,
        )
        return

    print(
        f"Embedding corpus={corpus_version}, model={embedding_model}, "
        f"documents={len(texts):,}, output={artifact_dir}",
        flush=True,
    )
    embedding_dim, elapsed = embed_texts_to_checkpointed_file(
        embed_model,
        texts,
        artifact_dir=artifact_dir,
        corpus_version=corpus_version,
        selected_sources=selected_sources,
        embedding_model=embedding_model,
        batch_size=CORPUS_BATCH_SIZE,
        progress_label=f"{embedding_model} {corpus_version}",
    )
    _write_manifest(
        corpus_version=corpus_version,
        selected_sources=selected_sources,
        document_count=len(records),
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        artifact_dir=artifact_dir,
        elapsed_seconds=elapsed,
    )
    checkpoint_path = _checkpoint_path(artifact_dir)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(
        f"Finished corpus={corpus_version}, model={embedding_model}, "
        f"shape=({len(records)}, {embedding_dim}), manifest={artifact_dir / 'manifest.json'}",
        flush=True,
    )


def main() -> None:
    ensure_data_directories()
    if EMBEDDING_BACKEND not in LOCAL_EMBEDDING_BACKENDS:
        raise RuntimeError(f"Unexpected local embedding backend: {EMBEDDING_BACKEND}")
    for embedding_model in LOCAL_BGE_MODELS:
        embed_model = _load_huggingface_embedding_model(embedding_model)
        for corpus_version in CORPUS_VERSIONS_TO_EMBED:
            embed_corpus_version(
                embed_model,
                embedding_model=embedding_model,
                corpus_version=corpus_version,
            )


if __name__ == "__main__":
    main()
