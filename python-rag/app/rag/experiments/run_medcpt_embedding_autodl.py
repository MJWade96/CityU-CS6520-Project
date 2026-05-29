"""Generate reusable MedCPT chunk embeddings on AutoDL.

This script intentionally reuses the MedCPT-specific parts of
``docs/embed_medcpt_corpus.py``: SentenceTransformer with CLS pooling and
``[title, content]`` article inputs. Project-specific corpus loading and output
paths stay here so the main RAG pipeline remains API-only.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

from app.rag.data.corpus_registry import combine_registered_corpora
from app.rag.data.data_paths import RESULT_INDEXES_DIR, ensure_data_directories
from app.rag.data.json_utils import save_json_atomic
from app.rag.experiments.phase1_formal_ablation import CORPUS_VARIANTS, FAISS_INDEX_TYPE


CORPUS_VERSIONS_TO_EMBED = ("statpearls", "statpearls_textbooks")
FORMAL_MEDCPT_MODEL = "ncbi/MedCPT"
MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
EMBEDDING_BACKEND = "local_medcpt"
BATCH_SIZE = 64
SHOW_PROGRESS_BAR = True
SOURCE_RUNTIME = "autodl"
EMBEDDING_INPUT_FORMAT = "title_content_pair"


class CustomizeSentenceTransformer(SentenceTransformer):
    """Use MedScore's CLS pooling setup instead of SentenceTransformer mean pooling."""

    def _load_auto_model(self, model_name_or_path: str, *args: Any, **kwargs: Any) -> List[Any]:
        token = kwargs.get("token", None)
        cache_folder = kwargs.get("cache_folder", None)
        revision = kwargs.get("revision", None)
        trust_remote_code = kwargs.get("trust_remote_code", False)

        if (
            "token" in kwargs
            or "cache_folder" in kwargs
            or "revision" in kwargs
            or "trust_remote_code" in kwargs
        ):
            transformer_model = Transformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args={
                    "token": token,
                    "trust_remote_code": trust_remote_code,
                    "revision": revision,
                },
                tokenizer_args={
                    "token": token,
                    "trust_remote_code": trust_remote_code,
                    "revision": revision,
                },
            )
        else:
            transformer_model = Transformer(model_name_or_path)

        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), "cls")
        return [transformer_model, pooling_model]


def _slug(value: Any) -> str:
    return str(value).lower().replace("/", "_").replace("\\", "_").replace(".", "p")


def _artifact_dir(corpus_version: str) -> Path:
    return (
        RESULT_INDEXES_DIR
        / f"{_slug(corpus_version)}__{_slug(FORMAL_MEDCPT_MODEL)}__{FAISS_INDEX_TYPE}"
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
            "content": str(record.get("content") or "").strip(),
            "source": str(record.get("source") or "unknown"),
        }
        for index, record in enumerate(result["records"])
        if str(record.get("content") or "").strip()
    ]
    if not records:
        raise ValueError(f"No embeddable corpus records found for {corpus_version}")
    return records, selected_sources


def _format_medcpt_article_inputs(records: Sequence[Mapping[str, Any]]) -> List[List[str]]:
    return [[str(record["title"]), str(record["content"])] for record in records]


def _load_medcpt_article_model() -> CustomizeSentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CustomizeSentenceTransformer(MEDCPT_ARTICLE_MODEL, device=device)
    model.eval()
    return model


def _write_manifest(
    *,
    corpus_version: str,
    selected_sources: Sequence[str],
    document_count: int,
    embedding_dim: int,
    artifact_dir: Path,
    elapsed_seconds: float,
) -> None:
    save_json_atomic(
        artifact_dir / "manifest.json",
        {
            "corpus_version": corpus_version,
            "selected_sources": list(selected_sources),
            "document_count": document_count,
            "embedding_model": FORMAL_MEDCPT_MODEL,
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
    model: CustomizeSentenceTransformer,
    corpus_version: str,
) -> None:
    records, selected_sources = _load_corpus_records(corpus_version)
    artifact_dir = _artifact_dir(corpus_version)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Embedding corpus={corpus_version}, documents={len(records):,}, "
        f"output={artifact_dir}",
        flush=True,
    )
    started_at = time.time()
    embeddings = model.encode(
        _format_medcpt_article_inputs(records),
        batch_size=BATCH_SIZE,
        show_progress_bar=SHOW_PROGRESS_BAR,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    np.save(artifact_dir / "chunk_embeddings.npy", embeddings)
    _write_manifest(
        corpus_version=corpus_version,
        selected_sources=selected_sources,
        document_count=len(records),
        embedding_dim=int(embeddings.shape[1]),
        artifact_dir=artifact_dir,
        elapsed_seconds=time.time() - started_at,
    )
    print(
        f"Finished corpus={corpus_version}, shape={embeddings.shape}, "
        f"manifest={artifact_dir / 'manifest.json'}",
        flush=True,
    )


def main() -> None:
    ensure_data_directories()
    model = _load_medcpt_article_model()
    with torch.no_grad():
        for corpus_version in CORPUS_VERSIONS_TO_EMBED:
            embed_corpus_version(model, corpus_version)


if __name__ == "__main__":
    main()
