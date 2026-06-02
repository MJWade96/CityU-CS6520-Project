"""Formal local rerank cache helpers.

The formal evaluator and the AutoDL rerank script share this module so cache
validation and LlamaIndex node conversion stay in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from llama_index.core.schema import NodeWithScore, TextNode

from app.rag.evaluation import formal_artifacts


LOCAL_RERANKER_BACKEND = "local_hf_cross_encoder"
RERANK_OUTPUTS_FILENAME = "rerank_outputs.jsonl"
LOCAL_RERANK_SCRIPT = "python -m app.rag.experiments.run_local_rerank_cache_autodl"


def _slug(value: Any) -> str:
    return str(value).lower().replace("/", "_").replace(" ", "_").replace(".", "p")


def rerank_cache_id(
    *,
    retrieval_candidates_id: str,
    reranker_model: str,
    reranker_input_count: int,
) -> str:
    """Cache key required by cache_design.md for reusable rerank outputs."""
    return (
        f"{_slug(retrieval_candidates_id)}"
        f"__{_slug(reranker_model)}"
        f"__input{int(reranker_input_count)}"
    )


def candidate_nodes(candidates: Sequence[Mapping[str, Any]]) -> List[NodeWithScore]:
    """Convert formal fusion candidate rows into LlamaIndex reranker inputs."""
    nodes: List[NodeWithScore] = []
    for candidate in candidates:
        nodes.append(
            NodeWithScore(
                node=TextNode(
                    text=str(candidate["text"]),
                    metadata=dict(candidate.get("metadata") or {}),
                ),
                score=float(candidate.get("score", 0.0)),
            )
        )
    return nodes


def rerank_rows_by_question(
    rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Index rerank output rows by question id and reject duplicate cache rows."""
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if not question_id:
            raise ValueError("rerank cache row is missing question_id")
        if question_id in indexed:
            raise ValueError(f"duplicate rerank cache row for question_id={question_id}")
        indexed[question_id] = dict(row)
    return indexed


def require_local_rerank_cache(
    path: Path,
    *,
    expected_question_ids: Sequence[str],
    expected_model: str,
) -> Dict[str, Dict[str, Any]]:
    """Load a complete local rerank cache or fail before any API rerank can run."""
    if not path.exists():
        raise FileNotFoundError(
            f"Formal local rerank cache is missing: {path}. "
            f"Copy the retrieval cache to AutoDL and run {LOCAL_RERANK_SCRIPT}."
        )

    rows = rerank_rows_by_question(formal_artifacts.load_jsonl(path))
    missing = [question_id for question_id in expected_question_ids if question_id not in rows]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"Formal local rerank cache {path} is incomplete; missing "
            f"{len(missing)} question(s), first: {preview}. "
            f"Regenerate it on AutoDL with {LOCAL_RERANK_SCRIPT}."
        )

    for question_id in expected_question_ids:
        row = rows[question_id]
        if row.get("reranker_backend") != LOCAL_RERANKER_BACKEND:
            raise RuntimeError(
                f"Rerank cache row {question_id} was not produced by "
                f"{LOCAL_RERANKER_BACKEND}: {row.get('reranker_backend')}"
            )
        if row.get("reranker_model") != expected_model:
            raise RuntimeError(
                f"Rerank cache row {question_id} uses model "
                f"{row.get('reranker_model')}, expected {expected_model}"
            )
        if not row.get("reranked_candidates"):
            raise RuntimeError(f"Rerank cache row {question_id} has no candidates")
    return rows
