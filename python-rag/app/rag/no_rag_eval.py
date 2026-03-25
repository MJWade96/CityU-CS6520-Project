"""
Baseline evaluation without retrieval augmentation.

This module reuses the shared evaluation helpers so the no-RAG and naive-RAG
pipelines stay aligned on dataset splitting, answer extraction, and rate limits.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data_paths import EVALUATION_DIR, EVALUATION_RESULTS_DIR
from .eval_shared import (
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
from .progress_manager import EvaluationProgressManager


@dataclass
class NoRAGEvalConfig:
    dev_size: int = 300
    test_size: Optional[int] = None
    question_file: Path = EVALUATION_DIR / "medqa.json"
    output_dir: Path = EVALUATION_RESULTS_DIR
    llm: EvaluationLLMConfig = field(default_factory=EvaluationLLMConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


async def evaluate_without_rag(
    config: NoRAGEvalConfig,
    progress_mgr: EvaluationProgressManager,
    artifact_paths: Dict[str, Path],
    start_from: int = 0,
    initial_results: Optional[List[Dict[str, Any]]] = None,
    initial_correct: int = 0,
    initial_elapsed: float = 0.0,
    script_name: str = "evaluate_no_rag",
) -> Dict[str, Any]:
    """Evaluate the model directly on the test set without retrieval."""
    questions = load_questions(str(config.question_file))
    _, test_set = split_questions(questions, config.dev_size, config.test_size)

    client = create_async_client(config.llm)
    semaphore = asyncio.Semaphore(config.concurrency.max_concurrent)
    rate_limiter = RateLimiter(
        requests_per_second=config.concurrency.requests_per_second,
        burst=config.concurrency.max_concurrent,
    )

    async def evaluate_item(item: Dict[str, Any]) -> Dict[str, Any]:
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
        return {
            "question": item["question"],
            "options": item.get("options", []),
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer == correct_answer,
            "response": response_content,
            "error": None,
        }

    start_time = time.time() - initial_elapsed
    results: List[Dict[str, Any]] = list(initial_results or [])
    correct = initial_correct
    config_payload = {
        "dev_set_size": config.dev_size,
        "test_set_size": len(test_set),
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.model,
        "evaluation_type": "NO_RAG",
    }
    remaining_questions = test_set[start_from:]
    batch_size = max(1, config.concurrency.max_concurrent)

    for batch_start in range(0, len(remaining_questions), batch_size):
        batch = remaining_questions[batch_start : batch_start + batch_size]
        batch_results = await asyncio.gather(*(evaluate_item(item) for item in batch))

        for offset, result in enumerate(batch_results, start=1):
            processed_questions = start_from + batch_start + offset
            results.append(result)
            if result["is_correct"]:
                correct += 1

            elapsed = time.time() - start_time
            progress_mgr.save_checkpoint(
                dataset_name="Test Set",
                total_questions=len(test_set),
                processed_questions=processed_questions,
                current_top_k=0,
                results=results,
                correct_count=correct,
                total_count=processed_questions,
                elapsed_time=elapsed,
                config=config_payload,
                script_name=script_name,
            )
            stage_result = progress_mgr.build_stage_result(
                dataset_name="Test Set",
                total_questions=len(test_set),
                processed_questions=processed_questions,
                correct_count=correct,
                elapsed_time=elapsed,
                detailed_results=results,
            )
            progress_mgr.print_progress(
                run_name="NO_RAG",
                dataset_name="Test Set",
                processed_questions=processed_questions,
                total_questions=len(test_set),
                correct_count=correct,
                elapsed_time=elapsed,
            )
            progress_mgr.write_live_results(
                artifact_paths=artifact_paths,
                run_name="NO_RAG",
                evaluation_type="NO_RAG",
                config=config_payload,
                stage_result=stage_result,
                extra_sections={"evaluation_results": stage_result},
            )

    elapsed = time.time() - start_time
    payload = {
        "dataset_name": "Test Set",
        "total_questions": len(test_set),
        "processed_questions": len(test_set),
        "correct": correct,
        "accuracy": correct / len(test_set) if test_set else 0.0,
        "elapsed_time": elapsed,
        "questions_per_second": len(test_set) / elapsed if elapsed > 0 else 0.0,
        "detailed_results": results,
    }
    return payload

async def run_no_rag_evaluation(config: NoRAGEvalConfig) -> Dict[str, Any]:
    progress_mgr = EvaluationProgressManager(output_dir=str(config.output_dir))
    artifact_paths = progress_mgr.create_run_artifacts("no_rag_eval")
    resume_test = progress_mgr.should_resume("evaluate_no_rag")
    resume_info_test = (
        progress_mgr.get_resume_info("evaluate_no_rag") if resume_test else None
    )
    results = await evaluate_without_rag(
        config,
        progress_mgr,
        artifact_paths,
        start_from=resume_info_test["start_from"] if resume_info_test else 0,
        initial_results=resume_info_test["results"] if resume_info_test else None,
        initial_correct=resume_info_test["correct_count"] if resume_info_test else 0,
        initial_elapsed=resume_info_test["elapsed_time"] if resume_info_test else 0.0,
        script_name="evaluate_no_rag",
    )
    progress_mgr.clear_checkpoint("evaluate_no_rag")
    paths = progress_mgr.write_final_results(
        artifact_paths=artifact_paths,
        run_name="NO_RAG",
        evaluation_type="NO_RAG",
        config={
            "dev_set_size": config.dev_size,
            "test_set_size": results["total_questions"],
            "llm_provider": config.llm.provider,
            "llm_model": config.llm.model,
        },
        stage_results={"evaluation_results": results},
    )
    return {"results": results, "output_paths": paths}
