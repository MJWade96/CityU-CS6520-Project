"""Generate rewritten query text cache for the final MedQA-USMLE test split."""

from __future__ import annotations

import asyncio

from app.rag.data.benchmarks.medqa_usmle import load_medqa_usmle_split
from app.rag.data.data_paths import ensure_data_directories
from app.rag.evaluation.eval_shared import EvaluationLLMConfig
from app.rag.experiments.run_local_bge_final_test_query_embedding_autodl import (
    DATASET_SPLIT,
    FINAL_TEST_QUERY_SPECS,
)


FINAL_TEST_REWRITE_SPECS = tuple(
    spec for spec in FINAL_TEST_QUERY_SPECS if spec.pipeline == "advanced_rag"
)


async def async_main() -> None:
    """Reuse the existing rewrite cache writer with final-test specs and split."""
    import app.rag.experiments.run_query_rewrite_cache_autodl as rewrite_cache

    ensure_data_directories()
    rewrite_cache.DATASET_SPLIT = DATASET_SPLIT
    questions = load_medqa_usmle_split(DATASET_SPLIT)
    await rewrite_cache.write_rewrite_caches(
        FINAL_TEST_REWRITE_SPECS,
        questions,
        EvaluationLLMConfig(),
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
