"""
Small-sample validation script to compare naive-RAG vs no-RAG before full evaluation.

This script runs a quick comparison on a small subset of questions to verify
that naive-RAG provides improvement over the no-RAG baseline before committing
to the full 973-question evaluation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from app.rag.data_paths import EVALUATION_DIR, EVALUATION_RESULTS_DIR, FAISS_INDEX_DIR
from app.rag.eval_shared import (
    ConcurrencyConfig,
    EvaluationLLMConfig,
    RateLimiter,
    build_medical_eval_prompt,
    create_async_client,
    extract_answer,
    get_qwen_completion_kwargs,
    get_correct_answer_letter,
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
    llm: EvaluationLLMConfig = field(default_factory=EvaluationLLMConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


async def evaluate_no_rag_sample(
    questions: List[Dict[str, Any]],
    config: SampleEvalConfig,
    progress_mgr: EvaluationProgressManager,
    artifact_paths: Dict[str, Path],
    live_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate without retrieval on a small sample."""
    client = create_async_client(config.llm)
    semaphore = asyncio.Semaphore(config.concurrency.max_concurrent)
    rate_limiter = RateLimiter(
        requests_per_second=config.concurrency.requests_per_second,
        burst=config.concurrency.max_concurrent,
    )

    results: List[Dict[str, Any]] = []
    correct_count = 0
    start_time = time.time()

    for idx, item in enumerate(questions):
        async with semaphore:
            await rate_limiter.acquire()
            completion = await client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": build_medical_eval_prompt(
                            question=item["question"],
                            options=item.get("options", []),
                        ),
                    }
                ],
                **get_qwen_completion_kwargs(config.llm),
            )

        response_content = (
            completion.choices[0].message.content
            or completion.choices[0].message.reasoning_content
            or ""
        )
        predicted_answer = extract_answer(response_content)
        correct_answer = get_correct_answer_letter(item)
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct_count += 1

        results.append(
            {
                "question": item["question"],
                "options": item.get("options", []),
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response": response_content,
            }
        )

        elapsed = time.time() - start_time
        progress_mgr.print_progress(
            run_name="NO_RAG",
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
        )
        progress_mgr.write_live_results(
            artifact_paths=artifact_paths,
            run_name="SAMPLE_VALIDATION",
            evaluation_type="NO_RAG",
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
    )


async def evaluate_naive_rag_sample(
    questions: List[Dict[str, Any]],
    vectorstore,
    config: SampleEvalConfig,
    progress_mgr: EvaluationProgressManager,
    artifact_paths: Dict[str, Path],
    live_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate with naive RAG on a small sample."""
    client = create_async_client(config.llm)
    semaphore = asyncio.Semaphore(config.concurrency.max_concurrent)
    rate_limiter = RateLimiter(
        requests_per_second=config.concurrency.requests_per_second,
        burst=config.concurrency.max_concurrent,
    )

    results: List[Dict[str, Any]] = []
    correct_count = 0
    start_time = time.time()

    for idx, item in enumerate(questions):
        search_results = await asyncio.to_thread(
            vectorstore.similarity_search_with_score,
            item["question"],
            config.top_k,
        )
        docs = [doc for doc, _ in search_results]
        scores = [float(score) for _, score in search_results]
        contexts = [doc.page_content for doc in docs]
        prompt = build_medical_eval_prompt(
            question=item["question"],
            options=item.get("options", []),
            context="\n\n".join(f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)),
        )

        async with semaphore:
            await rate_limiter.acquire()
            completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **get_qwen_completion_kwargs(config.llm),
            )

        response_content = (
            completion.choices[0].message.content
            or completion.choices[0].message.reasoning_content
            or ""
        )
        predicted_answer = extract_answer(response_content)
        correct_answer = get_correct_answer_letter(item)
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct_count += 1

        results.append(
            {
                "question": item["question"],
                "options": item.get("options", []),
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response": response_content,
                "retrieved_docs": len(docs),
                "scores": scores,
                "contexts": contexts,
            }
        )

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
            top_k=config.top_k,
        )
        progress_mgr.write_live_results(
            artifact_paths=artifact_paths,
            run_name="SAMPLE_VALIDATION",
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
        top_k=config.top_k,
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
    }

    print("=" * 60)
    print("Small-Sample Validation: NO_RAG vs NAIVE_RAG")
    print("=" * 60)
    print(f"Sample size: {config.sample_size}")
    print(f"Top-k for RAG: {config.top_k}")
    print(f"LLM model: {config.llm.model}")
    print()

    print("Running NO_RAG evaluation...")
    no_rag_results = await evaluate_no_rag_sample(
        sample_questions, config, progress_mgr, artifact_paths, live_config
    )
    print(
        f"  Accuracy: {no_rag_results['accuracy']:.4f} "
        f"({no_rag_results['correct']}/{no_rag_results['total_questions']})"
    )
    print(f"  Time: {no_rag_results['elapsed_time']:.1f}s")
    print()

    print("Loading vector store for NAIVE_RAG...")
    try:
        vectorstore = load_vector_store(config.vector_store_path)
        print("  Vector store loaded successfully")
    except Exception as e:
        print(f"  ERROR: Failed to load vector store: {e}")
        print("  Skipping NAIVE_RAG evaluation")
        progress_mgr.write_final_results(
            artifact_paths=artifact_paths,
            run_name="SAMPLE_VALIDATION",
            evaluation_type="COMPARISON",
            config=live_config,
            stage_results={"no_rag_evaluation": no_rag_results},
            extra_sections={"error": str(e), "naive_rag_evaluation": None},
        )
        return {"error": str(e), "no_rag_results": no_rag_results}

    print("Running NAIVE_RAG evaluation...")
    naive_rag_results = await evaluate_naive_rag_sample(
        sample_questions, vectorstore, config, progress_mgr, artifact_paths, live_config
    )
    print(
        f"  Accuracy: {naive_rag_results['accuracy']:.4f} "
        f"({naive_rag_results['correct']}/{naive_rag_results['total_questions']})"
    )
    print(f"  Time: {naive_rag_results['elapsed_time']:.1f}s")
    print()

    improvement = naive_rag_results["accuracy"] - no_rag_results["accuracy"]
    improvement_pct = (
        (improvement / no_rag_results["accuracy"] * 100)
        if no_rag_results["accuracy"] > 0
        else 0
    )

    print("=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(
        f"NO_RAG accuracy:    {no_rag_results['accuracy']:.4f} "
        f"({no_rag_results['correct']}/{no_rag_results['total_questions']})"
    )
    print(
        f"NAIVE_RAG accuracy: {naive_rag_results['accuracy']:.4f} "
        f"({naive_rag_results['correct']}/{naive_rag_results['total_questions']})"
    )
    print(f"Improvement:        {improvement:+.4f} ({improvement_pct:+.1f}%)")
    print()

    if improvement > 0:
        print("Result: NAIVE_RAG shows improvement over NO_RAG baseline.")
        print("Recommendation: Proceed with full evaluation.")
    elif improvement == 0:
        print("Result: NAIVE_RAG shows no improvement over NO_RAG baseline.")
        print("Recommendation: Review retrieval quality and prompt design.")
    else:
        print("Result: NAIVE_RAG performs worse than NO_RAG baseline.")
        print(
            "Recommendation: Investigate retrieval relevance and context integration."
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
