"""Run the final MedQA-USMLE test comparison for selected RAG configurations."""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from app.rag.data.json_utils import save_json_atomic
from app.rag.experiments.formal_ablation_executor import (
    FormalExecutionConfig,
    execute_formal_run,
)
from app.rag.experiments.phase1_formal_ablation import (
    FAISS_INDEX_TYPE,
    GENERATOR_MODEL,
    PROMPT_VERSION,
    RANDOM_SEED,
    FormalRunSpec,
)


FINAL_TEST_RUN_ID = "final_test_comparison"
DATASET_SPLIT = "test"
CORPUS_VERSION = "statpearls_textbooks"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_BACKEND = "local_hf_embedding"


FINAL_TEST_ROWS = (
    FormalRunSpec(
        stage=FINAL_TEST_RUN_ID,
        run_id="final_test_naive_bge_large_k10",
        pipeline="naive_rag",
        corpus_version=CORPUS_VERSION,
        embedding_model=EMBEDDING_MODEL,
        embedding_backend=EMBEDDING_BACKEND,
        faiss_index_type=FAISS_INDEX_TYPE,
        k=10,
        alpha=None,
        reranker_input_count=0,
        reranker_output_count=0,
        query_enhancement_setting="off",
        generator_model=GENERATOR_MODEL,
        prompt_version=PROMPT_VERSION,
        dataset_split=DATASET_SPLIT,
        random_seed=RANDOM_SEED,
    ),
    FormalRunSpec(
        stage=FINAL_TEST_RUN_ID,
        run_id="final_test_advanced_bge_large_k10_alpha0p5_rerank20",
        pipeline="advanced_rag",
        corpus_version=CORPUS_VERSION,
        embedding_model=EMBEDDING_MODEL,
        embedding_backend=EMBEDDING_BACKEND,
        faiss_index_type=FAISS_INDEX_TYPE,
        k=10,
        alpha=0.5,
        reranker_input_count=20,
        reranker_output_count=10,
        query_enhancement_setting="on",
        generator_model=GENERATOR_MODEL,
        prompt_version=PROMPT_VERSION,
        dataset_split=DATASET_SPLIT,
        random_seed=RANDOM_SEED,
    ),
)


async def run_final_test_comparison() -> dict:
    """Reuse formal run execution while constraining the run set to final test rows."""
    config = FormalExecutionConfig(dataset_split=DATASET_SPLIT)
    run_metrics = []
    for row in FINAL_TEST_ROWS:
        print(f"[final-test] run: {row.run_id}", flush=True)
        run_metrics.append(await execute_formal_run(row, config))
    return {
        "run_id": FINAL_TEST_RUN_ID,
        "status": "completed",
        "dataset_split": DATASET_SPLIT,
        "rows": [asdict(row) for row in FINAL_TEST_ROWS],
        "run_metrics": run_metrics,
    }


def main() -> None:
    manifest = asyncio.run(run_final_test_comparison())
    from app.rag.data.data_paths import RUNS_DIR

    output_path = RUNS_DIR / FINAL_TEST_RUN_ID / "execution_manifest.json"
    save_json_atomic(output_path, manifest, indent=2, ensure_ascii=False)
    print("=" * 60)
    print("Final Test Comparison Complete")
    print("=" * 60)
    print(f"Status: {manifest['status']}")
    print(f"Runs: {len(manifest['run_metrics'])}")
    print(f"Execution manifest: {output_path}")


if __name__ == "__main__":
    main()
