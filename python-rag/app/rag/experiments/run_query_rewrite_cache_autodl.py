"""Generate reusable rewritten query text cache on AutoDL.

This script only resolves advanced retrieval query texts. It writes the
``query_texts.jsonl`` file consumed later by ``run_medcpt_query_embedding_autodl.py``
and does not load MedCPT, run retrieval, rerank, build FAISS, or prompt the final
answering LLM.
"""

from __future__ import annotations

import asyncio
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
    _query_texts_path,
    _validate_query_text_rows,
    _write_jsonl,
)
from app.rag.retriever.query_rewrite import QueryRewritePipeline


REWRITE_PROGRESS_EVERY = 10
REWRITE_CACHE_IDS: Sequence[str] = ("advanced_medcpt_rewritten_query",)


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


async def build_advanced_query_text_rows(
    questions: Sequence[Mapping[str, Any]],
    llm_config: EvaluationLLMConfig,
) -> List[Dict[str, Any]]:
    """Run one shared query enhancement path for reusable advanced caches."""
    query_rewriter = QueryRewritePipeline(
        use_dict=True,
        use_llm=True,
        llm_provider=llm_config.provider,
        llm_model=llm_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        llm_temperature=llm_config.temperature,
        llm_enable_thinking=llm_config.enable_thinking,
    )
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
            {
                "question_id": item.get("id", f"{DATASET_SPLIT}-{index}"),
                "question": str(item["question"]),
                "original_query": original_query,
                "query_text": rewritten_query,
                "query_text_source": "query_rewrite_pipeline",
                "contains_options": False,
                "contains_answer_prompt": False,
            }
        )
        if (
            index == 1
            or index % REWRITE_PROGRESS_EVERY == 0
            or index == len(questions)
        ):
            print(f"  rewritten {index:,}/{len(questions):,} queries", flush=True)
    return rows


async def write_rewrite_cache(
    spec: QueryEmbeddingSpec,
    questions: Sequence[Mapping[str, Any]],
    llm_config: EvaluationLLMConfig,
) -> None:
    print(
        f"Rewriting cache={spec.cache_id}, questions={len(questions):,}",
        flush=True,
    )
    rows = await build_advanced_query_text_rows(questions, llm_config)
    _validate_query_text_rows(spec, rows, questions)
    output_path = _query_texts_path(spec)
    _write_jsonl(output_path, rows)
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
