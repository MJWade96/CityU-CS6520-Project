"""
Naive RAG generation script - reads cached retrieval results and generates answers.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from app.rag.eval_shared import (
    build_medical_eval_prompt,
    call_llm,
    ConcurrencyConfig,
    create_eval_context,
    EvaluationLLMConfig,
    extract_answer,
    get_correct_answer_letter,
)
from app.rag.progress_manager import EvaluationProgressManager
from naive_rag_shared import (
    CACHE_DIR,
    load_cached_retrieval,
    load_sample_questions,
    OUTPUT_DIR,
    QUESTION_FILE,
    run_tracked_workers,
    SAMPLE_SIZE,
    TOP_K,
    DEV_SIZE,
    write_live_sample_progress,
    WorkerResult,
)


@dataclass
class GenerationConfig:
    sample_size: int = SAMPLE_SIZE
    top_k: int = TOP_K
    dev_size: int = DEV_SIZE
    question_file: Path = QUESTION_FILE
    output_dir: Path = OUTPUT_DIR
    cache_dir: Path = CACHE_DIR
    llm: EvaluationLLMConfig = field(
        default_factory=lambda: EvaluationLLMConfig(enable_thinking=True)
    )
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


async def run_generation(config: GenerationConfig) -> Dict[str, Any]:
    """Run generation using cached retrieval results."""
    sample_questions = load_sample_questions(
        config.question_file, config.dev_size, config.sample_size
    )

    progress_mgr = EvaluationProgressManager(output_dir=str(config.output_dir))
    artifact_paths = progress_mgr.create_run_artifacts("naive_rag_generation")
    live_config = {
        "sample_size": config.sample_size,
        "top_k": config.top_k,
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.model,
        "enable_thinking": config.llm.enable_thinking,
    }

    print("=" * 60 + "\nNaive RAG Generation\n" + "=" * 60)
    print(f"Sample size: {config.sample_size}\nTop-k: {config.top_k}\nLLM model: {config.llm.model}\n")

    ctx = create_eval_context(config.llm, config.concurrency)

    start_time = time.time()

    async def process_item(item: Dict[str, Any]) -> WorkerResult[Dict[str, Any]]:
        question = item.get("question", "")
        cached = load_cached_retrieval(question)

        if not cached:
            print(f"  Warning: No cached retrieval for question: {question[:50]}...")
            result = {
                "question": question,
                "options": item.get("options", []),
                "correct_answer": get_correct_answer_letter(item),
                "predicted_answer": None,
                "is_correct": False,
                "error": "No cached retrieval found",
            }
            return WorkerResult(payload=result)

        contexts = cached.get("contexts", [])
        context_str = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        prompt = build_medical_eval_prompt(
            question=question,
            options=item.get("options", []),
            context=context_str,
        )

        response_content = await call_llm(ctx, prompt)
        predicted_answer = extract_answer(response_content)
        correct_answer = get_correct_answer_letter(item)

        result = {
            "question": question,
            "options": item.get("options", []),
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer == correct_answer,
            "response": response_content,
            "retrieved_docs": cached.get("retrieved_docs", 0),
            "scores": cached.get("scores", []),
        }
        return WorkerResult(payload=result, increment_correct=result["is_correct"])

    def handle_error(
        item: Dict[str, Any], error: Exception
    ) -> WorkerResult[Dict[str, Any]]:
        print(f"  Warning: Failed to generate answer: {error}")
        traceback.print_exc()
        return WorkerResult(
            payload={
                "question": item.get("question", ""),
                "options": item.get("options", []),
                "correct_answer": get_correct_answer_letter(item),
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
                artifact_run_name="NAIVE_RAG_GENERATION",
                evaluation_type="NAIVE_RAG",
                total_questions=len(sample_questions),
                processed_questions=processed,
                correct_count=correct_count,
                elapsed=elapsed,
                results=results,
                top_k=config.top_k,
            )
        except Exception as io_err:
            print(f"  Warning: Failed to save progress/artifacts: {io_err}")

    print("Running generation...")
    results, correct_count = await run_tracked_workers(
        items=sample_questions,
        process_item=process_item,
        max_concurrent=config.concurrency.max_concurrent,
        on_progress=handle_progress,
        on_error=handle_error,
    )

    elapsed = time.time() - start_time
    results_data = progress_mgr.build_stage_result(
        dataset_name="Sample",
        total_questions=len(sample_questions),
        processed_questions=len(sample_questions),
        correct_count=correct_count,
        elapsed_time=elapsed,
        detailed_results=results,
        top_k=config.top_k,
    )

    print(f"  Accuracy: {results_data['accuracy']:.4f} ({results_data['correct']}/{results_data['total_questions']})")
    print(f"  Time: {results_data['elapsed_time']:.1f}s\n")

    paths = progress_mgr.write_final_results(
        artifact_paths=artifact_paths,
        run_name="NAIVE_RAG_GENERATION",
        evaluation_type="NAIVE_RAG",
        config=live_config,
        stage_results={"sample_evaluation": results_data},
        extra_sections={
            "top_k": config.top_k,
            "sample_size": config.sample_size,
        },
    )
    print(f"\nResults saved to: {paths['json']}")
    return {
        "results": results_data,
        "output_paths": paths,
    }


async def main() -> None:
    config = GenerationConfig()
    await run_generation(config)


if __name__ == "__main__":
    asyncio.run(main())
