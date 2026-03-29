"""
Small-sample validation script to compare naive-RAG vs no-RAG before full evaluation.
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
    EvalContext,
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
class SampleEvalConfig:
    sample_size: int = SAMPLE_SIZE
    top_k: int = TOP_K
    dev_size: int = DEV_SIZE
    question_file: Path = QUESTION_FILE
    output_dir: Path = OUTPUT_DIR
    vector_store_path: Path = VECTOR_STORE_PATH
    llm: EvaluationLLMConfig = field(
        default_factory=lambda: EvaluationLLMConfig(enable_thinking=None)
    )
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


async def evaluate_sample(
    questions: List[Dict[str, Any]],
    ctx: EvalContext,
    config: SampleEvalConfig,
    progress_mgr: EvaluationProgressManager,
    artifact_paths: Dict[str, Path],
    live_config: Dict[str, Any],
    run_name: str,
    vectorstore: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate a sample of questions using shared evaluation context."""
    results: List[Dict[str, Any]] = []
    correct_count = 0
    start_time = time.time()

    for idx, item in enumerate(questions):
        result = await evaluate_single_item(ctx, item, vectorstore, config.top_k)
        results.append(result)
        if result["is_correct"]:
            correct_count += 1

        elapsed = time.time() - start_time
        progress_mgr.print_progress(
            run_name=run_name,
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
            top_k=config.top_k if vectorstore else None,
        )
        progress_mgr.write_live_results(
            artifact_paths=artifact_paths,
            run_name="SAMPLE_VALIDATION",
            evaluation_type=run_name,
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
        top_k=config.top_k if vectorstore else None,
    )


async def run_sample_comparison(config: SampleEvalConfig) -> Dict[str, Any]:
    """Run comparison between no-RAG and naive-RAG on a small sample."""
    all_questions = load_questions(str(config.question_file))
    _, test_questions = split_questions(all_questions, config.dev_size, None)
    sample_questions = test_questions[: config.sample_size]

    progress_mgr = EvaluationProgressManager(output_dir=str(config.output_dir))
    artifact_paths = progress_mgr.create_run_artifacts("sample_validation")
    live_config = {
        "sample_size": config.sample_size,
        "top_k": config.top_k,
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.model,
        "vector_store": str(config.vector_store_path),
        "no_rag_enable_thinking": False,
        "naive_rag_enable_thinking": True,
    }

    base_llm_config = {
        "provider": config.llm.provider,
        "model": config.llm.model,
        "temperature": config.llm.temperature,
        "base_url": config.llm.base_url,
        "api_key": config.llm.api_key,
    }
    no_rag_llm = EvaluationLLMConfig(**base_llm_config, enable_thinking=False)
    naive_rag_llm = EvaluationLLMConfig(**base_llm_config, enable_thinking=True)
    no_rag_ctx = create_eval_context(no_rag_llm, config.concurrency)
    naive_rag_ctx = create_eval_context(naive_rag_llm, config.concurrency)

    print("=" * 60 + "\nSmall-Sample Validation: NO_RAG vs NAIVE_RAG\n" + "=" * 60)
    print(
        f"Sample size: {config.sample_size}\nTop-k for RAG: {config.top_k}\nLLM model: {config.llm.model}\n"
    )

    print("Running NO_RAG evaluation (thinking disabled)...")
    no_rag_results = await evaluate_sample(
        sample_questions,
        no_rag_ctx,
        config,
        progress_mgr,
        artifact_paths,
        live_config,
        "NO_RAG",
    )
    print(
        f"  Accuracy: {no_rag_results['accuracy']:.4f} ({no_rag_results['correct']}/{no_rag_results['total_questions']})"
    )
    print(f"  Time: {no_rag_results['elapsed_time']:.1f}s\n")

    print("Loading vector store for NAIVE_RAG...")
    try:
        vectorstore = load_vector_store(config.vector_store_path)
        print("  Vector store loaded successfully")
    except Exception as e:
        print(f"  ERROR: Failed to load vector store: {e}")
        progress_mgr.write_final_results(
            artifact_paths=artifact_paths,
            run_name="SAMPLE_VALIDATION",
            evaluation_type="COMPARISON",
            config=live_config,
            stage_results={"no_rag_evaluation": no_rag_results},
            extra_sections={"error": str(e), "naive_rag_evaluation": None},
        )
        return {"error": str(e), "no_rag_results": no_rag_results}

    print("Running NAIVE_RAG evaluation (thinking enabled)...")
    naive_rag_results = await evaluate_sample(
        sample_questions,
        naive_rag_ctx,
        config,
        progress_mgr,
        artifact_paths,
        live_config,
        "NAIVE_RAG",
        vectorstore,
    )
    print(
        f"  Accuracy: {naive_rag_results['accuracy']:.4f} ({naive_rag_results['correct']}/{naive_rag_results['total_questions']})"
    )
    print(f"  Time: {naive_rag_results['elapsed_time']:.1f}s\n")

    improvement = naive_rag_results["accuracy"] - no_rag_results["accuracy"]
    improvement_pct = (
        (improvement / no_rag_results["accuracy"] * 100)
        if no_rag_results["accuracy"] > 0
        else 0
    )

    print("=" * 60 + "\nComparison Summary\n" + "=" * 60)
    print(
        f"NO_RAG accuracy:    {no_rag_results['accuracy']:.4f} ({no_rag_results['correct']}/{no_rag_results['total_questions']})"
    )
    print(
        f"NAIVE_RAG accuracy: {naive_rag_results['accuracy']:.4f} ({naive_rag_results['correct']}/{naive_rag_results['total_questions']})"
    )
    print(f"Improvement:        {improvement:+.4f} ({improvement_pct:+.1f}%)\n")

    if improvement > 0:
        print(
            "Result: NAIVE_RAG shows improvement over NO_RAG baseline.\nRecommendation: Proceed with full evaluation."
        )
    elif improvement == 0:
        print(
            "Result: NAIVE_RAG shows no improvement over NO_RAG baseline.\nRecommendation: Review retrieval quality and prompt design."
        )
    else:
        print(
            "Result: NAIVE_RAG performs worse than NO_RAG baseline.\nRecommendation: Investigate retrieval relevance and context integration."
        )

    paths = progress_mgr.write_final_results(
        artifact_paths=artifact_paths,
        run_name="SAMPLE_VALIDATION",
        evaluation_type="COMPARISON",
        config=live_config,
        stage_results={
            "no_rag_evaluation": no_rag_results,
            "naive_rag_evaluation": naive_rag_results,
        },
        extra_sections={
            "comparison": {
                "improvement": improvement,
                "improvement_percent": improvement_pct,
                "naive_rag_better": improvement > 0,
            }
        },
    )
    print(f"\nResults saved to: {paths['json']}")
    return {
        "no_rag_results": no_rag_results,
        "naive_rag_results": naive_rag_results,
        "comparison": {
            "improvement": improvement,
            "improvement_percent": improvement_pct,
        },
        "output_paths": paths,
    }


async def main() -> None:
    config = SampleEvalConfig()
    await run_sample_comparison(config)


if __name__ == "__main__":
    asyncio.run(main())