"""
Baseline evaluation without retrieval augmentation.

This module reuses the shared evaluation helpers so the no-RAG and naive-RAG
pipelines stay aligned on dataset splitting, answer extraction, and rate limits.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .eval_shared import (
    ConcurrencyConfig,
    EvaluationLLMConfig,
    RateLimiter,
    create_async_client,
    extract_answer,
    format_options,
    get_correct_answer_letter,
    load_questions,
    split_questions,
)


NO_RAG_PROMPT = """You are a medical expert assistant. Answer the following question based on your medical knowledge.

Question: {question}

Options:
{options}

Provide your answer in the following format:
Answer: [A/B/C/D/E]

Your response:"""


@dataclass
class NoRAGEvalConfig:
    dev_size: int = 50
    test_size: int = 50
    question_file: Path = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "medqa.json"
    output_dir: Path = Path(__file__).resolve().parents[2] / "results" / "evaluation"
    llm: EvaluationLLMConfig = field(default_factory=EvaluationLLMConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


async def evaluate_without_rag(config: NoRAGEvalConfig) -> Dict[str, Any]:
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
                model=config.llm.model,
                messages=[
                    {
                        "role": "user",
                        "content": NO_RAG_PROMPT.format(
                            question=item["question"],
                            options=format_options(item.get("options", [])),
                        ),
                    }
                ],
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
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
    tasks = [evaluate_item(item) for item in test_set]
    for future in asyncio.as_completed(tasks):
        result = await future
        results.append(result)
        if result["is_correct"]:
            correct += 1

    elapsed = time.time() - start_time
    payload = {
        "total_questions": len(test_set),
        "correct": correct,
        "accuracy": correct / len(test_set) if test_set else 0.0,
        "elapsed_time": elapsed,
        "questions_per_second": len(test_set) / elapsed if elapsed > 0 else 0.0,
        "detailed_results": results,
    }
    return payload


def save_results(config: NoRAGEvalConfig, results: Dict[str, Any]) -> Dict[str, Path]:
    """Persist baseline evaluation artifacts."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = config.output_dir / f"no_rag_eval_{timestamp}.json"
    summary_path = config.output_dir / f"no_rag_summary_{timestamp}.txt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {
                    "dev_set_size": config.dev_size,
                    "test_set_size": config.test_size,
                    "llm_provider": config.llm.provider,
                    "llm_model": config.llm.model,
                    "evaluation_type": "NO_RAG",
                },
                "evaluation_results": results,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Medical RAG System - Baseline Evaluation (WITHOUT RAG)\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"LLM: {config.llm.provider}/{config.llm.model}\n")
        handle.write(f"Dev Set Size: {config.dev_size}\n")
        handle.write(f"Test Set Size: {config.test_size}\n\n")
        handle.write("Test Set Results:\n")
        handle.write(f"  Total Questions: {results['total_questions']}\n")
        handle.write(f"  Correct Answers: {results['correct']}\n")
        handle.write(f"  Accuracy: {results['accuracy']:.4f}\n")
        handle.write(f"  Time: {results['elapsed_time']:.1f}s\n")
        handle.write(f"  Speed: {results['questions_per_second']:.2f} q/s\n")

    return {"json": json_path, "summary": summary_path}


async def run_no_rag_evaluation(config: NoRAGEvalConfig) -> Dict[str, Any]:
    results = await evaluate_without_rag(config)
    paths = save_results(config, results)
    return {"results": results, "output_paths": paths}
