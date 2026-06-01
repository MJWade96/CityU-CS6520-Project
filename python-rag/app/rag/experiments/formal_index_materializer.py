"""Materialize vector-store paths required by formal ablation rows."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable

from app.rag.data.corpus_registry import combine_registered_corpora
from app.rag.data.data_paths import RESULT_INDEXES_DIR, ensure_data_directories
from app.rag.data.json_utils import load_json_safe, save_json_atomic
from app.rag.data.medical_corpus.build_vector_index import IndexBuildConfig, build_index
from app.rag.experiments.phase1_formal_ablation import (
    CORPUS_VARIANTS,
    FAISS_INDEX_TYPE,
    FormalRunSpec,
    _slug,
)


API_INDEX_REQUIRED_FILES = (
    "metadata.json",
    "build_metadata.json",
    "docstore.json",
    "index_store.json",
    "default__vector_store.json",
)
FORMAL_API_EMBEDDING_BATCH_SIZE = 64
FORMAL_API_EMBEDDING_NUM_WORKERS = 4
FORMAL_API_INDEX_USE_ASYNC = True


def _index_root(row: FormalRunSpec) -> Path:
    embedding_slug = _slug(row.embedding_model or "unresolved_embedding")
    return RESULT_INDEXES_DIR / f"{_slug(row.corpus_version)}__{embedding_slug}__{row.faiss_index_type}"


def _index_files(index_dir: Path) -> Iterable[str]:
    if not index_dir.exists():
        return ()
    return sorted(path.name for path in index_dir.iterdir() if path.is_file())


def _has_api_index(index_dir: Path) -> bool:
    return all((index_dir / name).exists() for name in API_INDEX_REQUIRED_FILES)


def _write_index_manifest(
    row: FormalRunSpec,
    index_dir: Path,
    *,
    status: str,
    build_metadata: Dict[str, Any] | None = None,
) -> None:
    payload = {
        "artifact_id": index_dir.name,
        "artifact_group": "index",
        "status": status,
        "corpus_version": row.corpus_version,
        "embedding_model": row.embedding_model,
        "embedding_backend": row.embedding_backend,
        "faiss_index_type": row.faiss_index_type,
        "files": list(_index_files(index_dir)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z") if status == "completed" else None,
    }
    if build_metadata:
        payload["build_metadata"] = build_metadata
    save_json_atomic(index_dir / "manifest.json", payload, indent=2, ensure_ascii=False)


def _materialize_corpus_file(row: FormalRunSpec, index_dir: Path) -> Path:
    if row.corpus_version not in CORPUS_VARIANTS:
        known = ", ".join(sorted(CORPUS_VARIANTS))
        raise KeyError(f"Unknown corpus version {row.corpus_version!r}; expected one of {known}")

    corpus_file = index_dir / "input_corpus.json"
    if corpus_file.exists():
        return corpus_file

    selected_sources = CORPUS_VARIANTS[row.corpus_version]
    combined = combine_registered_corpora(selected_sources=selected_sources)
    save_json_atomic(corpus_file, combined["records"], indent=None, ensure_ascii=False)
    return corpus_file


def _ensure_api_index(row: FormalRunSpec, index_dir: Path) -> Path:
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists() and _has_api_index(index_dir):
        manifest = load_json_safe(manifest_path)
        if manifest.get("status") == "completed":
            return index_dir

    if _has_api_index(index_dir):
        _write_index_manifest(row, index_dir, status="completed")
        return index_dir

    checkpoint_path = index_dir / "build_checkpoint.json"
    partial_files = [
        name
        for name in _index_files(index_dir)
        if name not in {"input_corpus.json", "manifest.json", "build_checkpoint.json"}
    ]
    if partial_files and not checkpoint_path.exists():
        raise RuntimeError(
            f"Partial formal index exists without checkpoint in {index_dir}: {partial_files}"
        )

    corpus_file = _materialize_corpus_file(row, index_dir)
    build_metadata = build_index(
        IndexBuildConfig(
            corpus_file=corpus_file,
            index_dir=index_dir,
            embedding_model=str(row.embedding_model),
            corpus_version=row.corpus_version,
            faiss_index_type=row.faiss_index_type,
            batch_size=FORMAL_API_EMBEDDING_BATCH_SIZE,
            embedding_api_num_workers=FORMAL_API_EMBEDDING_NUM_WORKERS,
            index_use_async=FORMAL_API_INDEX_USE_ASYNC,
        )
    )
    _write_index_manifest(row, index_dir, status="completed", build_metadata=build_metadata)
    return index_dir


def _ensure_medcpt_artifact(row: FormalRunSpec, index_dir: Path) -> Path:
    required = (index_dir / "chunk_embeddings.npy", index_dir / "manifest.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "MedCPT formal rows require precomputed AutoDL artifacts. "
            f"Missing: {', '.join(missing)}"
        )
    return index_dir


def ensure_formal_index(row: FormalRunSpec) -> Path:
    """Return the index/artifact root for one resolved formal run."""
    ensure_data_directories()
    if row.embedding_model is None or row.embedding_backend is None:
        raise ValueError(f"Formal row is unresolved and cannot materialize an index: {row.run_id}")

    index_dir = _index_root(row)
    index_dir.mkdir(parents=True, exist_ok=True)

    if row.embedding_backend == "local_medcpt":
        return _ensure_medcpt_artifact(row, index_dir)
    if row.embedding_backend == "siliconflow_api":
        return _ensure_api_index(row, index_dir)
    raise ValueError(f"Unsupported embedding backend for formal index: {row.embedding_backend}")
