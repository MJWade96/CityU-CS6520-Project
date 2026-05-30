"""Generate reusable MedCPT query embeddings on AutoDL.

This script only performs query embedding. Naive query texts are derived from
the MedQA-USMLE question field; advanced rewritten query texts must already be
stored by ``run_query_rewrite_cache_autodl.py``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from app.rag.data.benchmarks.medqa_usmle import load_medqa_usmle_split
from app.rag.data.data_paths import RETRIEVAL_CACHE_DIR, ensure_data_directories
from app.rag.data.json_utils import save_json_atomic


DATASET_SPLIT = "dev"
FORMAL_MEDCPT_MODEL = "ncbi/MedCPT"
MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
EMBEDDING_BACKEND = "local_medcpt"
BATCH_SIZE = 2048
MEDCPT_QUERY_MAX_LENGTH = 64
QUERY_TEXTS_FILENAME = "query_texts.jsonl"
QUERY_EMBEDDING_MANIFEST_FILENAME = "query_embedding_manifest.json"
SOURCE_RUNTIME = "autodl"
QUERY_INPUT_FORMAT = "retrieval_query_text_only"


@dataclass(frozen=True)
class QueryEmbeddingSpec:
    """One MedCPT query embedding cache target independent of formal matrix rows."""

    cache_id: str
    pipeline: str
    query_text_source: str


QUERY_EMBEDDING_SPECS: Sequence[QueryEmbeddingSpec] = (
    QueryEmbeddingSpec(
        cache_id="stage1_naive_medcpt",
        pipeline="naive_rag",
        query_text_source="medqa_usmle_question_field",
    ),
    QueryEmbeddingSpec(
        cache_id="advanced_medcpt_rewritten_query",
        pipeline="advanced_rag",
        query_text_source="query_rewrite_pipeline",
    ),
)


def _query_texts_path(spec: QueryEmbeddingSpec) -> Path:
    return RETRIEVAL_CACHE_DIR / spec.cache_id / QUERY_TEXTS_FILENAME


def _query_embeddings_path(spec: QueryEmbeddingSpec) -> Path:
    return RETRIEVAL_CACHE_DIR / spec.cache_id / "query_embeddings.npy"


def _query_embedding_manifest_path(spec: QueryEmbeddingSpec) -> Path:
    return RETRIEVAL_CACHE_DIR / spec.cache_id / QUERY_EMBEDDING_MANIFEST_FILENAME


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def build_naive_query_text_rows(
    questions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep query text construction separate from final answer prompt formatting."""
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(questions, start=1):
        rows.append(
            {
                "question_id": item.get("id", f"{DATASET_SPLIT}-{index}"),
                "question": str(item["question"]),
                "query_text": str(item["question"]),
                "query_text_source": "medqa_usmle_question_field",
                "contains_options": False,
                "contains_answer_prompt": False,
            }
        )
    return rows


def _validate_query_text_rows(
    spec: QueryEmbeddingSpec,
    rows: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
) -> None:
    if len(rows) != len(questions):
        raise ValueError(
            f"{spec.cache_id} query text count {len(rows)} does not match "
            f"{DATASET_SPLIT} question count {len(questions)}"
        )
    for index, row in enumerate(rows, start=1):
        if not str(row.get("query_text") or "").strip():
            raise ValueError(f"{spec.cache_id} query_text is empty at row {index}")


def resolve_query_text_rows(
    spec: QueryEmbeddingSpec,
    questions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve embedding inputs without rewrite, retrieval, or answer generation."""
    path = _query_texts_path(spec)
    if spec.pipeline == "naive_rag":
        rows = build_naive_query_text_rows(questions)
        _write_jsonl(path, rows)
        return rows

    if spec.pipeline == "advanced_rag":
        if not path.exists():
            raise FileNotFoundError(
                f"{spec.cache_id} requires rewritten query texts at {path}. "
                "Run run_query_rewrite_cache_autodl.py before query embedding."
            )
        rows = list(_iter_jsonl(path))
        _validate_query_text_rows(spec, rows, questions)
        return [dict(row) for row in rows]

    raise ValueError(f"Unsupported pipeline for query embeddings: {spec.pipeline}")


def _load_medcpt_query_model() -> Any:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MEDCPT_QUERY_MODEL)
    model = AutoModel.from_pretrained(MEDCPT_QUERY_MODEL)
    model.eval()
    model.to(device)
    return tokenizer, model, device


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype("float32")


def embed_query_texts(
    tokenizer: Any,
    model: Any,
    device: str,
    texts: Sequence[str],
) -> np.ndarray:
    """Embed query texts through one MedCPT query encoder path."""
    import torch

    vectors: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = list(texts[start : start + BATCH_SIZE])
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=MEDCPT_QUERY_MAX_LENGTH,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            embeddings = model(**encoded).last_hidden_state[:, 0, :].detach().cpu().numpy()
            vectors.append(embeddings.astype("float32"))
            print(
                f"  MedCPT query embedded {min(start + len(batch), len(texts)):,}/"
                f"{len(texts):,} texts",
                flush=True,
            )
    return _normalize_rows(np.vstack(vectors))


def write_query_embedding_manifest(
    spec: QueryEmbeddingSpec,
    *,
    query_text_count: int,
    embedding_dim: int,
    elapsed_seconds: float,
) -> None:
    save_json_atomic(
        _query_embedding_manifest_path(spec),
        {
            "cache": asdict(spec),
            "dataset_split": DATASET_SPLIT,
            "query_text_count": query_text_count,
            "embedding_model": FORMAL_MEDCPT_MODEL,
            "query_encoder_model": MEDCPT_QUERY_MODEL,
            "embedding_backend": EMBEDDING_BACKEND,
            "query_input_format": QUERY_INPUT_FORMAT,
            "contains_options": False,
            "contains_answer_prompt": False,
            "query_texts_path": str(_query_texts_path(spec)),
            "query_embeddings_path": str(_query_embeddings_path(spec)),
            "source_runtime": SOURCE_RUNTIME,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "build_time_seconds": elapsed_seconds,
            "embedding_dim": embedding_dim,
        },
    )


def embed_run_queries(
    spec: QueryEmbeddingSpec,
    questions: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    device: str,
) -> None:
    rows = resolve_query_text_rows(spec, questions)
    texts = [str(row["query_text"]) for row in rows]
    output_path = _query_embeddings_path(spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Embedding cache={spec.cache_id}, pipeline={spec.pipeline}, "
        f"queries={len(texts):,}, output={output_path}",
        flush=True,
    )
    started_at = time.time()
    embeddings = embed_query_texts(tokenizer, model, device, texts)
    np.save(output_path, embeddings)
    write_query_embedding_manifest(
        spec,
        query_text_count=len(texts),
        embedding_dim=int(embeddings.shape[1]),
        elapsed_seconds=time.time() - started_at,
    )
    print(
        f"Finished cache={spec.cache_id}, shape={embeddings.shape}, "
        f"manifest={_query_embedding_manifest_path(spec)}",
        flush=True,
    )


def main() -> None:
    ensure_data_directories()
    questions = load_medqa_usmle_split(DATASET_SPLIT)
    tokenizer, model, device = _load_medcpt_query_model()
    for spec in QUERY_EMBEDDING_SPECS:
        embed_run_queries(
            spec,
            questions,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )


if __name__ == "__main__":
    main()
