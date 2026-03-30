"""
Naive RAG generation script - reads cached retrieval results and generates answers.
"""

from __future__ import annotations

import asyncio
import time
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
    SAMPLE_SIZE,
    TOP_K,
    DEV_SIZE,
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

    results: List[Dict[str, Any]] = []
    correct_count = 0
    start_time = time.time()
    processed = 0
    lock = asyncio.Lock()

    async def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
        question = item.get("question", "")
        cached = load_cached_retrieval(question)

        if not cached:
            print(f"  Warning: No cached retrieval for question: {question[:50]}...")
            return {
                "question": question,
                "options": item.get("options", []),
                "correct_answer": get_correct_answer_letter(item),
                "predicted_answer": None,
                "is_correct": False,
                "error": "No cached retrieval found",
            }

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

        return {
            "question": question,
            "options": item.get("options", []),
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer == correct_answer,
            "response": response_content,
            "retrieved_docs": cached.get("retrieved_docs", 0),
            "scores": cached.get("scores", []),
        }

    print("Running generation...")
    queue: asyncio.Queue[Dict[str, Any] | None] = asyncio.Queue()
    for item in sample_questions:
        await queue.put(item)

    async def worker() -> None:
        nonlocal correct_count, processed
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            try:
                result = await process_item(item)
                async with lock:
                    results.append(result)
                    if result["is_correct"]:
                        correct_count += 1
                    processed += 1
                    current_processed = processed
                    current_correct = correct_count

                elapsed = time.time() - start_time
                progress_mgr.print_progress(
                    run_name="GENERATION",
                    dataset_name="Sample",
                    processed_questions=current_processed,
                    total_questions=len(sample_questions),
                    correct_count=current_correct,
                    elapsed_time=elapsed,
                )

                stage_result = progress_mgr.build_stage_result(
                    dataset_name="Sample",
                    total_questions=len(sample_questions),
                    processed_questions=current_processed,
                    correct_count=current_correct,
                    elapsed_time=elapsed,
                    detailed_results=results,
                    top_k=config.top_k,
                )
                progress_mgr.write_live_results(
                    artifact_paths=artifact_paths,
                    run_name="NAIVE_RAG_GENERATION",
                    evaluation_type="NAIVE_RAG",
                    config=live_config,
                    stage_result=stage_result,
                    status="running",
                )
            except Exception as e:
                print(f"  Error generating answer: {e}")
            finally:
                queue.task_done()

    workers = [
        asyncio.create_task(worker()) for _ in range(config.concurrency.max_concurrent)
    ]
    await queue.join()

    for _ in range(config.concurrency.max_concurrent):
        await queue.put(None)
    await asyncio.gather(*workers, return_exceptions=True)

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
