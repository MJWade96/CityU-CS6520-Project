"""
Naive RAG sample validation script - evaluates a small sample of questions using RAG.
"""

from __future__ import annotations
import traceback
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.rag.data.data_paths import EVALUATION_RESULTS_DIR, FAISS_INDEX_DIR, MEDQA_FILE
from app.rag.evaluation.eval_shared import (
    ConcurrencyConfig,
    create_eval_context,
    evaluate_single_item,
    EvaluationLLMConfig,
    load_questions,
    split_questions,
)
from app.rag.evaluation.naive_rag_eval import load_vector_store
from app.rag.utils.progress_manager import EvaluationProgressManager
from naive_rag_shared import (
    run_tracked_workers,
    write_live_sample_progress,
    WorkerResult,
)


SAMPLE_SIZE = 973
TOP_K = 3
DEV_SIZE = 300
QUESTION_FILE = MEDQA_FILE
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
    concurrency: ConcurrencyConfig,
) -> Dict[str, Any]:
    """Evaluate a sample of questions using naive RAG with concurrent execution."""
    start_time = time.time()

    async def process_item(item: Dict[str, Any]) -> WorkerResult[Dict[str, Any]]:
        result = await evaluate_single_item(ctx, item, vectorstore, top_k)
        return WorkerResult(payload=result, increment_correct=result["is_correct"])

    def handle_error(
        item: Dict[str, Any], error: Exception
    ) -> WorkerResult[Dict[str, Any]]:
        print(f"  Warning: Failed to evaluate question: {error}")
        traceback.print_exc()
        return WorkerResult(
            payload={
                "question": item.get("question", ""),
                "options": item.get("options", []),
                "correct_answer": "",
                "predicted_answer": "",
                "is_correct": False,
                "error": str(error),
            }
        )

    def handle_progress(
        results: List[Dict[str, Any]], processed: int, correct_count: int, elapsed: float
    ) -> None:
        try:
            write_live_sample_progress(
                progress_mgr,
                artifact_paths,
                live_config,
                console_run_name="NAIVE_RAG",
                artifact_run_name="NAIVE_RAG_SAMPLE",
                evaluation_type="NAIVE_RAG",
                total_questions=len(questions),
                processed_questions=processed,
                correct_count=correct_count,
                elapsed=elapsed,
                results=results,
                top_k=top_k,
            )
        except Exception as io_err:
            print(f"  Warning: Failed to save progress/artifacts: {io_err}")

    results, correct_count = await run_tracked_workers(
        items=questions,
        process_item=process_item,
        max_concurrent=concurrency.max_concurrent,
        on_progress=handle_progress,
        on_error=handle_error,
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
        config.concurrency,
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
