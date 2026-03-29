"""
Naive RAG sample validation script - evaluates a small sample of questions using RAG.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.rag.data_paths import EVALUATION_DIR, EVALUATION_RESULTS_DIR, FAISS_INDEX_DIR
from app.rag.eval_shared import (
    ConcurrencyConfig,
    create_eval_context,
    evaluate_single_item,
    EvaluationLLMConfig,
    load_questions,
    split_questions,
)
from app.rag.naive_rag_eval import load_vector_store
from app.rag.progress_manager import EvaluationProgressManager


SAMPLE_SIZE = 50
TOP_K = 3
DEV_SIZE = 0
QUESTION_FILE = EVALUATION_DIR / "medqa.json"
OUTPUT_DIR = EVALUATION_RESULTS_DIR
VECTOR_STORE_PATH = FAISS_INDEX_DIR


@dataclass
class NaiveRAGSevalConfig:
    sample_size: int = SAMPLE_SIZE
    top_k: int = TOP_K
    dev_size: int = DEV_SIZE
    question_file: Path = QUESTION_FILE
    output_dir: Path = OUTPUT_DIR
    vector_store_path: Path = VECTOR_STORE_PATH
    llm: EvaluationLLMConfig = field(
        default_factory=lambda: EvaluationLLMConfig(enable_thinking=True)
    )
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


async def evaluate_naive_rag_sample(
    questions: List[Dict[str, Any]],
    ctx,
    progress_mgr: EvaluationProgressManager,
    artifact_paths: Dict[str, Path],
    live_config: Dict[str, Any],
    vectorstore,
    top_k: int,
) -> Dict[str, Any]:
    """Evaluate a sample of questions using naive RAG."""
    results: List[Dict[str, Any]] = []
    correct_count = 0
    start_time = time.time()

    for idx, item in enumerate(questions):
        result = await evaluate_single_item(ctx, item, vectorstore, top_k)
        results.append(result)
        if result["is_correct"]:
            correct_count += 1

        elapsed = time.time() - start_time
        progress_mgr.print_progress(
            run_name="NAIVE_RAG",
            dataset_name="Sample",
            processed_questions=idx + 1,
            total_questions=len(questions),
            correct_count=correct_count,
            elapsed_time=elapsed,
        )
        stage_result = progress_mgr.build_stage_result(
            dataset_name="Sample",
            total_questions=len(questions),
            processed_questions=idx + 1,
            correct_count=correct_count,
            elapsed_time=elapsed,
            detailed_results=results,
            top_k=top_k,
        )
        progress_mgr.write_live_results(
            artifact_paths=artifact_paths,
            run_name="NAIVE_RAG_SAMPLE",
            evaluation_type="NAIVE_RAG",
            config=live_config,
            stage_result=stage_result,
            status="running",
        )

    elapsed = time.time() - start_time
    return progress_mgr.build_stage_result(
        dataset_name="Sample",
        total_questions=len(questions),
        processed_questions=len(questions),
        correct_count=correct_count,
        elapsed_time=elapsed,
        detailed_results=results,
        top_k=top_k,
    )


async def run_naive_rag_sample(config: NaiveRAGSevalConfig) -> Dict[str, Any]:
    """Run naive RAG evaluation on a small sample."""
    all_questions = load_questions(str(config.question_file))
    _, test_questions = split_questions(all_questions, config.dev_size, None)
    sample_questions = test_questions[: config.sample_size]

    progress_mgr = EvaluationProgressManager(output_dir=str(config.output_dir))
    artifact_paths = progress_mgr.create_run_artifacts("naive_rag_sample_eval")
    live_config = {
        "sample_size": config.sample_size,
        "top_k": config.top_k,
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.model,
        "vector_store": str(config.vector_store_path),
        "enable_thinking": config.llm.enable_thinking,
    }

    print("=" * 60 + "\nNaive RAG Sample Validation\n" + "=" * 60)
    print(
        f"Sample size: {config.sample_size}\nTop-k: {config.top_k}\nLLM model: {config.llm.model}\n"
    )

    print("Loading vector store...")
    try:
        vectorstore = load_vector_store(config.vector_store_path)
        print("  Vector store loaded successfully")
    except Exception as e:
        print(f"  ERROR: Failed to load vector store: {e}")
        progress_mgr.write_final_results(
            artifact_paths=artifact_paths,
            run_name="NAIVE_RAG_SAMPLE",
            evaluation_type="NAIVE_RAG",
            config=live_config,
            stage_results={},
            extra_sections={"error": str(e)},
        )
        return {"error": str(e)}

    ctx = create_eval_context(config.llm, config.concurrency)

    print("Running NAIVE_RAG evaluation...")
    results = await evaluate_naive_rag_sample(
        sample_questions,
        ctx,
        progress_mgr,
        artifact_paths,
        live_config,
        vectorstore,
        config.top_k,
    )
    print(
        f"  Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total_questions']})"
    )
    print(f"  Time: {results['elapsed_time']:.1f}s\n")

    paths = progress_mgr.write_final_results(
        artifact_paths=artifact_paths,
        run_name="NAIVE_RAG_SAMPLE",
        evaluation_type="NAIVE_RAG",
        config=live_config,
        stage_results={"sample_evaluation": results},
        extra_sections={
            "top_k": config.top_k,
            "sample_size": config.sample_size,
        },
    )
    print(f"\nResults saved to: {paths['json']}")
    return {
        "results": results,
        "output_paths": paths,
    }


async def main() -> None:
    config = NaiveRAGSevalConfig()
    await run_naive_rag_sample(config)


if __name__ == "__main__":
    asyncio.run(main())