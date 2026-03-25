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
from typing import Any, Dict, List

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
    dev_size: int = 50
    test_size: int = 50
    question_file: Path = EVALUATION_DIR / "medqa.json"
    output_dir: Path = EVALUATION_RESULTS_DIR
    llm: EvaluationLLMConfig = field(default_factory=EvaluationLLMConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


async def evaluate_without_rag(
    config: NoRAGEvalConfig,
    progress_mgr: EvaluationProgressManager,
    artifact_paths: Dict[str, Path],
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

    start_time = time.time()
    results: List[Dict[str, Any]] = []
    correct = 0
    config_payload = {
        "dev_set_size": config.dev_size,
        "test_set_size": config.test_size,
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.model,
        "evaluation_type": "NO_RAG",
    }
    tasks = [evaluate_item(item) for item in test_set]
    for index, future in enumerate(asyncio.as_completed(tasks), start=1):
        result = await future
        results.append(result)
        if result["is_correct"]:
            correct += 1

        elapsed = time.time() - start_time
        stage_result = progress_mgr.build_stage_result(
            dataset_name="Test Set",
            total_questions=len(test_set),
            processed_questions=index,
            correct_count=correct,
            elapsed_time=elapsed,
            detailed_results=results,
        )
        progress_mgr.print_progress(
            run_name="NO_RAG",
            dataset_name="Test Set",
            processed_questions=index,
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
    results = await evaluate_without_rag(config, progress_mgr, artifact_paths)
    paths = progress_mgr.write_final_results(
        artifact_paths=artifact_paths,
        run_name="NO_RAG",
        evaluation_type="NO_RAG",
        config={
            "dev_set_size": config.dev_size,
            "test_set_size": config.test_size,
            "llm_provider": config.llm.provider,
            "llm_model": config.llm.model,
        },
        stage_results={"evaluation_results": results},
    )
    return {"results": results, "output_paths": paths}
