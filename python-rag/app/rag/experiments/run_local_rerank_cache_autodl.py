"""Generate formal local rerank caches on AutoDL.

Run this after copying populated ``results/retrieval_cache`` directories from
the PC. The script scans caches with ``fusion_candidates.jsonl`` and writes the
matching ``results/rerank_cache/<cache_id>/rerank_outputs.jsonl`` artifacts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from llama_index.core.postprocessor import SentenceTransformerRerank

from app.rag.data.data_paths import (
    RERANK_CACHE_DIR,
    RETRIEVAL_CACHE_DIR,
    ensure_data_directories,
)
from app.rag.evaluation import formal_artifacts
from app.rag.evaluation.eval_shared import serialize_node_candidates
from app.rag.evaluation.formal_local_rerank_cache import (
    LOCAL_RERANKER_BACKEND,
    RERANK_OUTPUTS_FILENAME,
    candidate_nodes,
)
from app.rag.retriever.runtime_config import DEFAULT_API_RERANKER_MODEL


LOCAL_RERANKER_MODEL = DEFAULT_API_RERANKER_MODEL
DATASET_SPLIT = "dev"
SOURCE_RUNTIME = "autodl"
FUSION_CANDIDATES_FILENAME = "fusion_candidates.jsonl"
RERANK_DEVICE = "cuda"


def _fusion_candidate_paths() -> List[Path]:
    return sorted(RETRIEVAL_CACHE_DIR.glob(f"*/{FUSION_CANDIDATES_FILENAME}"))


def _max_candidate_count(rows: Sequence[Mapping[str, Any]]) -> int:
    counts = [len(row.get("candidates") or []) for row in rows]
    if not counts or max(counts) <= 0:
        raise RuntimeError("fusion_candidates rows must contain at least one candidate")
    return max(counts)


def _load_reranker(top_n: int) -> SentenceTransformerRerank:
    return SentenceTransformerRerank(
        model=LOCAL_RERANKER_MODEL,
        top_n=top_n,
        device=RERANK_DEVICE,
        keep_retrieval_score=True,
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def rerank_cache_rows(
    cache_id: str,
    fusion_rows: Sequence[Mapping[str, Any]],
    reranker: SentenceTransformerRerank,
) -> List[Dict[str, Any]]:
    """Build rerank cache rows through one local LlamaIndex postprocessor path."""
    rows: List[Dict[str, Any]] = []
    total = len(fusion_rows)
    for index, row in enumerate(fusion_rows, start=1):
        question_id = str(row["question_id"])
        query_text = str(row["query_text"])
        candidates = list(row.get("candidates") or [])
        started_at = time.time()
        reranked_nodes = reranker.postprocess_nodes(
            candidate_nodes(candidates),
            query_str=query_text,
        )
        elapsed = time.time() - started_at
        rows.append(
            {
                "question_id": question_id,
                "input_candidates_id": f"{question_id}:fusion_candidates",
                "reranker_backend": LOCAL_RERANKER_BACKEND,
                "reranker_model": LOCAL_RERANKER_MODEL,
                "reranker_input_count": len(candidates),
                "reranker_output_count": len(reranked_nodes),
                "reranked_candidates": serialize_node_candidates(reranked_nodes),
                "rerank_time_seconds": elapsed,
            }
        )
        if index == total or index % 10 == 0:
            print(
                f"  reranked cache={cache_id} {index:,}/{total:,} questions",
                flush=True,
            )
    return rows


def write_rerank_cache(fusion_path: Path) -> None:
    cache_id = fusion_path.parent.name
    output_dir = RERANK_CACHE_DIR / cache_id
    output_path = output_dir / RERANK_OUTPUTS_FILENAME
    fusion_rows = formal_artifacts.load_jsonl(fusion_path)
    top_n = _max_candidate_count(fusion_rows)
    print(
        f"Reranking cache={cache_id}, questions={len(fusion_rows):,}, "
        f"max_candidates={top_n}, output={output_path}",
        flush=True,
    )
    reranker = _load_reranker(top_n)
    started_at = time.time()
    rows = rerank_cache_rows(cache_id, fusion_rows, reranker)
    _write_jsonl(output_path, rows)
    formal_artifacts.write_json(
        output_dir / "manifest.json",
        {
            "cache_id": cache_id,
            "status": "completed",
            "pipeline": "advanced_rag",
            "dataset_split": DATASET_SPLIT,
            "reranker_backend": LOCAL_RERANKER_BACKEND,
            "reranker_model": LOCAL_RERANKER_MODEL,
            "source_runtime": SOURCE_RUNTIME,
            "processed_questions": len(rows),
            "input_artifacts": {"fusion_candidates": str(fusion_path)},
            "files": {"rerank_outputs": str(output_path)},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "build_time_seconds": time.time() - started_at,
        },
    )
    print(f"Finished rerank cache={cache_id}, manifest={output_dir / 'manifest.json'}")


def main() -> None:
    ensure_data_directories()
    fusion_paths = _fusion_candidate_paths()
    if not fusion_paths:
        raise FileNotFoundError(
            f"No {FUSION_CANDIDATES_FILENAME} files found under {RETRIEVAL_CACHE_DIR}. "
            "Run formal ablation on the PC until retrieval caches are written, then copy "
            "those caches to AutoDL."
        )
    for fusion_path in fusion_paths:
        write_rerank_cache(fusion_path)


if __name__ == "__main__":
    main()
