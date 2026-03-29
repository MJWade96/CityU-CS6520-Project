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
    create_eval_context,
    evaluate_single_item,
    EvaluationLLMConfig,
    load_questions,
    split_questions,
    update_progress,
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

    ctx = create_eval_context(config.llm, config.concurrency)
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

    async def evaluate_item(item: Dict[str, Any]) -> Dict[str, Any]:
        return await evaluate_single_item(ctx, item)

    for batch_start in range(0, len(remaining_questions), batch_size):
        batch = remaining_questions[batch_start : batch_start + batch_size]
        batch_results = await asyncio.gather(*(evaluate_item(item) for item in batch))

        for offset, result in enumerate(batch_results, start=1):
            processed_questions = start_from + batch_start + offset
            results.append(result)
            if result["is_correct"]:
                correct += 1

            update_progress(
                progress_mgr=progress_mgr,
                artifact_paths=artifact_paths,
                live_config=config_payload,
                extra_sections={"evaluation_results": {}},
                dataset_name="Test Set",
                total_questions=len(test_set),
                processed_questions=processed_questions,
                correct_count=correct,
                elapsed=time.time() - start_time,
                results=results,
                run_name="NO_RAG",
                evaluation_type="NO_RAG",
                config_payload=config_payload,
                script_name=script_name,
            )

    elapsed = time.time() - start_time
    return {
        "dataset_name": "Test Set",
        "total_questions": len(test_set),
        "processed_questions": len(test_set),
        "correct": correct,
        "accuracy": correct / len(test_set) if test_set else 0.0,
        "elapsed_time": elapsed,
        "questions_per_second": len(test_set) / elapsed if elapsed > 0 else 0.0,
        "detailed_results": results,
    }


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
