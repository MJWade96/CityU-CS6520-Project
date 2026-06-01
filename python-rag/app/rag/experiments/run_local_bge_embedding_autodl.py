"""Generate reusable local BGE chunk embeddings on AutoDL.

This script mirrors the formal API index input text while changing only the
embedding generation backend. It writes artifact arrays consumed later by the
formal evaluator on the PC.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from app.rag.data.corpus_registry import combine_registered_corpora
from app.rag.data.data_paths import RESULT_INDEXES_DIR, ensure_data_directories
from app.rag.data.json_utils import save_json_atomic
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
BATCH_SIZE = 256
SOURCE_RUNTIME = "autodl"
EMBEDDING_INPUT_FORMAT = "corpus_content_text"


def _artifact_dir(corpus_version: str, embedding_model: str):
    return (
        RESULT_INDEXES_DIR
        / f"{_slug(corpus_version)}__{_slug(embedding_model)}__{FAISS_INDEX_TYPE}"
    )


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
        embed_batch_size=BATCH_SIZE,
    )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype("float32")


def embed_texts(embed_model: Any, texts: Sequence[str]) -> np.ndarray:
    """Use one local embedding call path for corpus and query scripts."""
    vectors = embed_model.get_text_embedding_batch(list(texts), show_progress=True)
    return _normalize_rows(np.asarray(vectors, dtype="float32"))


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

    print(
        f"Embedding corpus={corpus_version}, model={embedding_model}, "
        f"documents={len(texts):,}, output={artifact_dir}",
        flush=True,
    )
    started_at = time.time()
    embeddings = embed_texts(embed_model, texts)
    np.save(artifact_dir / "chunk_embeddings.npy", embeddings)
    _write_manifest(
        corpus_version=corpus_version,
        selected_sources=selected_sources,
        document_count=len(records),
        embedding_model=embedding_model,
        embedding_dim=int(embeddings.shape[1]),
        artifact_dir=artifact_dir,
        elapsed_seconds=time.time() - started_at,
    )
    print(
        f"Finished corpus={corpus_version}, model={embedding_model}, "
        f"shape={embeddings.shape}, manifest={artifact_dir / 'manifest.json'}",
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
