"""Generate reusable rewritten query text cache on AutoDL.

This script only resolves advanced retrieval query texts. It writes the
``query_texts.jsonl`` file consumed later by ``run_medcpt_query_embedding_autodl.py``
and does not load MedCPT, run retrieval, rerank, build FAISS, or prompt the final
answering LLM.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Mapping, Sequence

from app.rag.data.benchmarks.medqa_usmle import load_medqa_usmle_split
from app.rag.data.data_paths import ensure_data_directories
from app.rag.evaluation.eval_shared import (
    ConcurrencyConfig,
    EvaluationLLMConfig,
    create_eval_context,
)
from app.rag.experiments.run_medcpt_query_embedding_autodl import (
    DATASET_SPLIT,
    QUERY_EMBEDDING_SPECS,
    QueryEmbeddingSpec,
    _iter_jsonl,
    _query_texts_path,
    _validate_query_text_rows,
    _write_jsonl,
)
from app.rag.retriever.query_rewrite import QueryRewritePipeline


REWRITE_PROGRESS_EVERY = 10
REWRITE_CACHE_IDS: Sequence[str] = ("advanced_medcpt_rewritten_query",)
RUN_MODE = "rewrite_all"  # "rewrite_all" or "retry_errors"
QUERY_TEXTS_CHECKPOINT_FILENAME = "query_texts.checkpoint.jsonl"
QUERY_REWRITE_ERRORS_FILENAME = "query_rewrite_errors.jsonl"


def _selected_rewrite_specs() -> List[QueryEmbeddingSpec]:
    specs_by_id = {spec.cache_id: spec for spec in QUERY_EMBEDDING_SPECS}
    selected: List[QueryEmbeddingSpec] = []
    for cache_id in REWRITE_CACHE_IDS:
        if cache_id not in specs_by_id:
            raise KeyError(f"Unknown query rewrite cache id: {cache_id}")
        spec = specs_by_id[cache_id]
        if spec.pipeline != "advanced_rag":
            raise ValueError(f"{cache_id} is not an advanced query rewrite cache")
        selected.append(spec)
    return selected


def _rewrite_checkpoint_path(spec: QueryEmbeddingSpec):
    return _query_texts_path(spec).with_name(QUERY_TEXTS_CHECKPOINT_FILENAME)


def _rewrite_errors_path(spec: QueryEmbeddingSpec):
    return _query_texts_path(spec).with_name(QUERY_REWRITE_ERRORS_FILENAME)


def _append_jsonl(path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _question_id(item: Mapping[str, Any], index: int) -> str:
    return str(item.get("id", f"{DATASET_SPLIT}-{index}"))


def _build_rewrite_text_row(
    item: Mapping[str, Any],
    *,
    index: int,
    original_query: str,
    rewritten_query: str,
) -> Dict[str, Any]:
    return {
        "question_id": _question_id(item, index),
        "question": str(item["question"]),
        "original_query": original_query,
        "query_text": rewritten_query,
        "query_text_source": "query_rewrite_pipeline",
        "contains_options": False,
        "contains_answer_prompt": False,
    }


def _load_existing_rows(spec: QueryEmbeddingSpec) -> List[Dict[str, Any]]:
    checkpoint_path = _rewrite_checkpoint_path(spec)
    output_path = _query_texts_path(spec)
    if checkpoint_path.exists():
        return [dict(row) for row in _iter_jsonl(checkpoint_path)]
    if output_path.exists():
        return [dict(row) for row in _iter_jsonl(output_path)]
    return []


def _load_error_rows(spec: QueryEmbeddingSpec) -> List[Dict[str, Any]]:
    errors_path = _rewrite_errors_path(spec)
    if not errors_path.exists():
        return []
    return [dict(row) for row in _iter_jsonl(errors_path)]


def _error_question_ids(spec: QueryEmbeddingSpec) -> set[str]:
    return {
        str(row.get("question_id", ""))
        for row in _load_error_rows(spec)
        if str(row.get("question_id", "")).strip()
    }


def _rows_by_question_id(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id", ""))
        if question_id in by_id:
            raise ValueError(f"Duplicate checkpoint question_id: {question_id}")
        by_id[question_id] = dict(row)
    return by_id


def _questions_by_id(
    questions: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    return {
        _question_id(item, index): item
        for index, item in enumerate(questions, start=1)
    }


def _question_order(questions: Sequence[Mapping[str, Any]]) -> List[str]:
    return [
        _question_id(item, index)
        for index, item in enumerate(questions, start=1)
    ]


def create_query_rewriter(llm_config: EvaluationLLMConfig) -> QueryRewritePipeline:
    """Keep rewrite client construction shared across fresh and resumed runs."""
    return QueryRewritePipeline(
        use_dict=True,
        use_llm=True,
        llm_provider=llm_config.provider,
        llm_model=llm_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        llm_temperature=llm_config.temperature,
        llm_enable_thinking=llm_config.enable_thinking,
    )


async def build_advanced_query_text_rows(
    questions: Sequence[Mapping[str, Any]],
    llm_config: EvaluationLLMConfig,
) -> List[Dict[str, Any]]:
    """Run one shared query enhancement path for reusable advanced caches."""
    query_rewriter = create_query_rewriter(llm_config)
    ctx = create_eval_context(llm_config, ConcurrencyConfig())
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(questions, start=1):
        original_query = str(item["question"])
        rewritten_query, _ = await query_rewriter.arewrite(
            original_query,
            rate_limiter=ctx.rate_limiter,
            api_semaphore=ctx.semaphore,
            use_llm=True,
        )
        rows.append(
            _build_rewrite_text_row(
                item,
                index=index,
                original_query=original_query,
                rewritten_query=rewritten_query,
            )
        )
        if (
            index == 1
            or index % REWRITE_PROGRESS_EVERY == 0
            or index == len(questions)
        ):
            print(f"  rewritten {index:,}/{len(questions):,} queries", flush=True)
    return rows


async def build_advanced_query_text_rows_with_checkpoint(
    spec: QueryEmbeddingSpec,
    questions: Sequence[Mapping[str, Any]],
    llm_config: EvaluationLLMConfig,
    *,
    retry_errors_only: bool = False,
) -> List[Dict[str, Any]]:
    """Append each successful rewrite and keep failed rows for a retry-only pass."""
    checkpoint_path = _rewrite_checkpoint_path(spec)
    errors_path = _rewrite_errors_path(spec)
    rows_by_id = _rows_by_question_id(_load_existing_rows(spec))
    question_order = _question_order(questions)
    questions_by_id = _questions_by_id(questions)
    if len(rows_by_id) == len(question_order):
        rows = [rows_by_id[question_id] for question_id in question_order]
        _validate_query_text_rows(spec, rows, questions)
        return rows

    failed_question_ids = _error_question_ids(spec)
    if retry_errors_only:
        pending_question_ids = [
            question_id
            for question_id in question_order
            if question_id in failed_question_ids and question_id not in rows_by_id
        ]
    else:
        pending_question_ids = [
            question_id
            for question_id in question_order
            if question_id not in rows_by_id and question_id not in failed_question_ids
        ]

    if retry_errors_only and not failed_question_ids:
        raise FileNotFoundError(
            f"No rewrite error rows found at {errors_path}; run rewrite_all first."
        )

    query_rewriter = create_query_rewriter(llm_config)
    ctx = create_eval_context(llm_config, ConcurrencyConfig())
    for question_id in pending_question_ids:
        item = questions_by_id[question_id]
        index = question_order.index(question_id) + 1
        original_query = str(item["question"])
        try:
            rewritten_query, _ = await query_rewriter.arewrite(
                original_query,
                rate_limiter=ctx.rate_limiter,
                api_semaphore=ctx.semaphore,
                use_llm=True,
            )
        except Exception as exc:
            _append_jsonl(
                errors_path,
                {
                    "question_id": question_id,
                    "question": str(item["question"]),
                    "original_query": original_query,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "run_mode": "retry_errors" if retry_errors_only else "rewrite_all",
                },
            )
            print(
                f"  failed rewriting {index:,}/{len(questions):,} queries, "
                f"question_id={question_id}, error_log={errors_path}",
                flush=True,
            )
            continue

        row = _build_rewrite_text_row(
            item,
            index=index,
            original_query=original_query,
            rewritten_query=rewritten_query,
        )
        _append_jsonl(checkpoint_path, row)
        rows_by_id[question_id] = row
        if (
            index == 1
            or index % REWRITE_PROGRESS_EVERY == 0
            or index == len(questions)
        ):
            print(f"  rewritten {index:,}/{len(questions):,} queries", flush=True)

    rows = [
        rows_by_id[question_id]
        for question_id in question_order
        if question_id in rows_by_id
    ]
    return rows


async def write_rewrite_cache(
    spec: QueryEmbeddingSpec,
    questions: Sequence[Mapping[str, Any]],
    llm_config: EvaluationLLMConfig,
) -> None:
    retry_errors_only = RUN_MODE == "retry_errors"
    if RUN_MODE not in {"rewrite_all", "retry_errors"}:
        raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")
    print(
        f"Rewriting cache={spec.cache_id}, mode={RUN_MODE}, questions={len(questions):,}",
        flush=True,
    )
    rows = await build_advanced_query_text_rows_with_checkpoint(
        spec,
        questions,
        llm_config,
        retry_errors_only=retry_errors_only,
    )
    if len(rows) != len(questions):
        missing = len(questions) - len(rows)
        print(
            f"Rewrite cache incomplete: completed={len(rows):,}, "
            f"missing={missing:,}, error_log={_rewrite_errors_path(spec)}",
            flush=True,
        )
        return

    _validate_query_text_rows(spec, rows, questions)
    output_path = _query_texts_path(spec)
    _write_jsonl(output_path, rows)
    checkpoint_path = _rewrite_checkpoint_path(spec)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Finished query rewrite cache={spec.cache_id}, output={output_path}", flush=True)


async def async_main() -> None:
    ensure_data_directories()
    questions = load_medqa_usmle_split(DATASET_SPLIT)
    llm_config = EvaluationLLMConfig()
    for spec in _selected_rewrite_specs():
        await write_rewrite_cache(spec, questions, llm_config)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
