"""Generate reusable local BGE query embeddings on AutoDL.

Naive query texts are derived from the MedQA-USMLE question field. Advanced
query texts must already exist because query rewriting is a separate stage.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from app.rag.data.benchmarks.medqa_usmle import load_medqa_usmle_split
from app.rag.data.data_paths import RETRIEVAL_CACHE_DIR, ensure_data_directories
from app.rag.data.json_utils import save_json_atomic
from app.rag.experiments.formal_cache_metadata import manifest_metadata, path_fingerprint
from app.rag.experiments.formal_query_embedding_specs import QueryEmbeddingSpec
from app.rag.experiments.phase1_formal_ablation import (
    EMBEDDING_PROVIDERS,
    LOCAL_EMBEDDING_BACKENDS,
    build_formal_matrix,
    _slug,
)
from app.rag.experiments.run_medcpt_query_embedding_autodl import (
    DATASET_SPLIT,
    QUERY_EMBEDDING_MANIFEST_FILENAME,
    QUERY_TEXTS_FILENAME,
    _iter_jsonl,
    _validate_query_text_rows,
    _write_jsonl,
    build_naive_query_text_rows,
)
from app.rag.experiments.run_local_bge_embedding_autodl import (
    SOURCE_RUNTIME,
    embed_texts,
)


EMBEDDING_BACKEND = "local_hf_embedding"
QUERY_INPUT_FORMAT = "retrieval_query_text_only"
QUERY_BATCH_SIZE = 256


def _local_bge_models() -> List[str]:
    return [
        provider.model
        for provider in EMBEDDING_PROVIDERS
        if provider.backend == EMBEDDING_BACKEND
    ]


def query_cache_id(run_id: str, embedding_model: str) -> str:
    return f"{run_id}__{_slug(embedding_model)}"


def build_bge_query_embedding_specs() -> List[QueryEmbeddingSpec]:
    """Generate all possible BGE query caches without depending on stage winners."""
    specs_by_id: Dict[str, QueryEmbeddingSpec] = {}
    local_models = _local_bge_models()
    for row in build_formal_matrix():
        if row.embedding_backend == EMBEDDING_BACKEND and row.embedding_model is not None:
            models = [row.embedding_model]
        elif row.selection_rule:
            models = local_models
        else:
            continue

        for embedding_model in models:
            source = (
                "query_rewrite_pipeline"
                if row.pipeline == "advanced_rag"
                else "medqa_usmle_question_field"
            )
            spec = QueryEmbeddingSpec(
                cache_id=query_cache_id(row.run_id, embedding_model),
                pipeline=row.pipeline,
                query_text_source=source,
            )
            specs_by_id[spec.cache_id] = spec
    return list(specs_by_id.values())


BGE_QUERY_EMBEDDING_SPECS: Sequence[QueryEmbeddingSpec] = tuple(
    build_bge_query_embedding_specs()
)


def _query_texts_path(spec: QueryEmbeddingSpec):
    return RETRIEVAL_CACHE_DIR / spec.cache_id / QUERY_TEXTS_FILENAME


def _query_embeddings_path(spec: QueryEmbeddingSpec):
    return RETRIEVAL_CACHE_DIR / spec.cache_id / "query_embeddings.npy"


def _query_embedding_manifest_path(spec: QueryEmbeddingSpec):
    return RETRIEVAL_CACHE_DIR / spec.cache_id / QUERY_EMBEDDING_MANIFEST_FILENAME


def resolve_query_text_rows(
    spec: QueryEmbeddingSpec,
    questions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve query embedding inputs while preserving the existing prompt boundary."""
    path = _query_texts_path(spec)
    if spec.pipeline == "naive_rag":
        rows = build_naive_query_text_rows(questions)
        _write_jsonl(path, rows)
        return rows

    if spec.pipeline == "advanced_rag":
        if not path.exists():
            raise FileNotFoundError(
                f"{spec.cache_id} requires rewritten query texts at {path}. "
                "Run run_query_rewrite_cache_autodl.py before local BGE query embedding."
            )
        rows = list(_iter_jsonl(path))
        _validate_query_text_rows(spec, rows, questions)
        return [dict(row) for row in rows]

    raise ValueError(f"Unsupported pipeline for query embeddings: {spec.pipeline}")


def _load_huggingface_embedding_model(model_name: str) -> Any:
    import torch
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        embed_batch_size=QUERY_BATCH_SIZE,
    )


def write_query_embedding_manifest(
    spec: QueryEmbeddingSpec,
    *,
    embedding_model: str,
    query_text_count: int,
    embedding_dim: int,
    elapsed_seconds: float,
) -> None:
    query_texts_path = _query_texts_path(spec)
    save_json_atomic(
        _query_embedding_manifest_path(spec),
        {
            "cache": asdict(spec),
            "dataset_split": DATASET_SPLIT,
            "query_text_count": query_text_count,
            "embedding_model": embedding_model,
            "embedding_backend": EMBEDDING_BACKEND,
            "query_input_format": QUERY_INPUT_FORMAT,
            "contains_options": False,
            "contains_answer_prompt": False,
            "query_texts_path": str(query_texts_path),
            "query_embeddings_path": str(_query_embeddings_path(spec)),
            "source_runtime": SOURCE_RUNTIME,
            "build_time_seconds": elapsed_seconds,
            "embedding_dim": embedding_dim,
            **manifest_metadata(
                key={
                    "cache_id": spec.cache_id,
                    "pipeline": spec.pipeline,
                    "query_text_source": spec.query_text_source,
                    "embedding_model": embedding_model,
                },
                input_artifacts={"query_texts": str(query_texts_path)},
                parameters={
                    "embedding_backend": EMBEDDING_BACKEND,
                    "query_input_format": QUERY_INPUT_FORMAT,
                    "batch_size": QUERY_BATCH_SIZE,
                },
                dataset_split=DATASET_SPLIT,
                fingerprint={
                    "query_texts": path_fingerprint(query_texts_path),
                    "query_text_count": query_text_count,
                },
            ),
        },
    )


def embed_run_queries(
    spec: QueryEmbeddingSpec,
    questions: Sequence[Mapping[str, Any]],
    *,
    embedding_model: str,
    embed_model: Any,
) -> None:
    rows = resolve_query_text_rows(spec, questions)
    texts = [str(row["query_text"]) for row in rows]
    output_path = _query_embeddings_path(spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Embedding cache={spec.cache_id}, model={embedding_model}, "
        f"pipeline={spec.pipeline}, queries={len(texts):,}, output={output_path}",
        flush=True,
    )
    started_at = time.time()
    embeddings = embed_texts(
        embed_model,
        texts,
        batch_size=QUERY_BATCH_SIZE,
        progress_label=f"{embedding_model} {spec.cache_id}",
    )
    np.save(output_path, embeddings)
    write_query_embedding_manifest(
        spec,
        embedding_model=embedding_model,
        query_text_count=len(texts),
        embedding_dim=int(embeddings.shape[1]),
        elapsed_seconds=time.time() - started_at,
    )
    print(
        f"Finished cache={spec.cache_id}, shape={embeddings.shape}, "
        f"manifest={_query_embedding_manifest_path(spec)}",
        flush=True,
    )


def _specs_for_model(embedding_model: str) -> Iterable[QueryEmbeddingSpec]:
    suffix = f"__{_slug(embedding_model)}"
    return [spec for spec in BGE_QUERY_EMBEDDING_SPECS if spec.cache_id.endswith(suffix)]


def main() -> None:
    ensure_data_directories()
    if EMBEDDING_BACKEND not in LOCAL_EMBEDDING_BACKENDS:
        raise RuntimeError(f"Unexpected local embedding backend: {EMBEDDING_BACKEND}")
    questions = load_medqa_usmle_split(DATASET_SPLIT)
    for embedding_model in _local_bge_models():
        embed_model = _load_huggingface_embedding_model(embedding_model)
        for spec in _specs_for_model(embedding_model):
            embed_run_queries(
                spec,
                questions,
                embedding_model=embedding_model,
                embed_model=embed_model,
            )


if __name__ == "__main__":
    main()
