"""
Performance comparison script for evaluating sync vs async LLM inference

This script tests both versions on a subset of questions to measure
performance improvement from async optimization.
"""

import os
import sys
import json
import time
import asyncio
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from openai import AsyncOpenAI


class EvalConfig:
    """Evaluation configuration"""

    SCRIPT_DIR = Path(__file__).parent

    LLM_PROVIDER = "deepseek"
    LLM_MODEL = "2656053fa69c4c2d89c5a691d9d737c3"
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 512
    LLM_BASE_URL = "https://wishub-x6.ctyun.cn/v1"
    LLM_API_KEY = "6fcecb364d0647d2883e7f1d3f19d5b9"

    QUESTION_FILE = str(SCRIPT_DIR / "data" / "evaluation" / "medqa.json")
    OUTPUT_DIR = str(SCRIPT_DIR / "results" / "evaluation")

    DEV_SET_SIZE = 100


NO_RAG_PROMPT = """You are a medical expert assistant. Answer the following question based on your medical knowledge.

Question: {question}

Options:
{options}

Please think step by step and then provide your answer in the following format:
Answer: [A/B/C/D/E]

Your response:"""


def load_questions(question_file: str) -> List[Dict]:
    """Load MedQA questions"""
    print(f"Loading questions from {question_file}...")

    if not os.path.exists(question_file):
        print(f"ERROR: File not found: {question_file}")
        return []

    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"[OK] Loaded {len(questions)} questions")
    return questions


def extract_answer(response: str) -> Optional[str]:
    """Extract answer choice from LLM response"""
    patterns = [
        r"Answer:\s*([A-E])",
        r"answer:\s*([A-E])",
        r"\b([A-E])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


async def get_response(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    item: Dict,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict:
    """Get response from LLM with semaphore-controlled concurrency"""
    async with semaphore:
        question_text = item.get("question", "")
        options = item.get("options", [])

        if options:
            options_text = "\n".join(
                [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
            )
        else:
            options_text = (
                "A. Not provided\nB. Not provided\nC. Not provided\nD. Not provided"
            )

        prompt = NO_RAG_PROMPT.format(
            question=question_text,
            options=options_text,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_content = completion.choices[0].message.content
            return {
                "question": question_text,
                "options": options,
                "response": response_content,
                "error": None,
            }
        except Exception as e:
            return {
                "question": question_text,
                "options": options,
                "response": None,
                "error": str(e),
            }


async def evaluate_async(
    questions: List[Dict],
    config: EvalConfig,
    max_concurrent: int = 20,
) -> Dict[str, Any]:
    """Evaluate questions using async LLM calls"""
    print(f"\n{'=' * 70}")
    print(f"Async Evaluation (max_concurrent={max_concurrent})")
    print(f"{'=' * 70}")

    client = AsyncOpenAI(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    start_time = time.time()

    tasks = [
        get_response(
            semaphore,
            client,
            q,
            config.LLM_MODEL,
            config.LLM_TEMPERATURE,
            config.LLM_MAX_TOKENS,
        )
        for q in questions
    ]

    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    correct = 0
    total = 0
    detailed_results = []

    for i, (q, result) in enumerate(zip(questions, results)):
        answer_index = q.get("answer_index", -1)
        correct_answer = q.get("answer", "")

        if answer_index >= 0:
            correct_answer_letter = chr(65 + answer_index)
        else:
            correct_answer_letter = correct_answer

        if result["error"]:
            predicted_answer = None
            is_correct = False
        else:
            predicted_answer = extract_answer(result["response"])
            is_correct = predicted_answer == correct_answer_letter.upper()

        detailed_results.append(
            {
                "question": result["question"],
                "options": result["options"],
                "correct_answer": correct_answer_letter,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response": result["response"],
                "error": result["error"],
            }
        )

        if is_correct:
            correct += 1
        total += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            qps = (i + 1) / elapsed if elapsed > 0 else 0
            current_acc = correct / total if total > 0 else 0
            print(
                f"  Question {i+1}/{len(questions)} | "
                f"Accuracy: {current_acc:.4f} | "
                f"Speed: {qps:.2f} q/s"
            )

    accuracy = correct / total if total > 0 else 0

    return {
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_time": elapsed,
        "questions_per_second": total / elapsed if elapsed > 0 else 0,
        "detailed_results": detailed_results,
    }


async def run_comparison(
    questions: List[Dict],
    subset_size: int = 50,
    concurrent_sizes: List[int] = None,
):
    """Run performance comparison with different concurrency levels"""
    if concurrent_sizes is None:
        concurrent_sizes = [10, 20, 30]

    print("=" * 70)
    print("Performance Comparison: Async LLM Inference")
    print("=" * 70)
    print(f"\nTesting on subset of {subset_size} questions")
    print(f"Concurrency levels to test: {concurrent_sizes}")
    print()

    test_questions = questions[:subset_size]
    config = EvalConfig()

    results = {}

    for max_concurrent in concurrent_sizes:
        print(f"\n{'=' * 70}")
        print(f"TEST: max_concurrent = {max_concurrent}")
        print(f"{'=' * 70}")

        try:
            result = await evaluate_async(
                test_questions,
                config,
                max_concurrent=max_concurrent,
            )
            results[f"concurrent_{max_concurrent}"] = {
                "elapsed": result["elapsed_time"],
                "results": result,
            }
            print(f"\n[OK] Completed with max_concurrent={max_concurrent}")
            print(f"  Time: {result['elapsed_time']:.2f} seconds")
            print(f"  Speed: {result['questions_per_second']:.2f} q/s")
            print(f"  Accuracy: {result['accuracy']:.4f}")
        except Exception as e:
            print(f"\n[ERROR] Failed with max_concurrent={max_concurrent}: {e}")
            results[f"concurrent_{max_concurrent}"] = {
                "elapsed": None,
                "results": None,
                "error": str(e),
            }

    print()
    print("=" * 70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 70)

    valid_results = {k: v for k, v in results.items() if v.get("results") is not None}

    if len(valid_results) >= 2:
        sorted_results = sorted(
            valid_results.items(),
            key=lambda x: x[1]["results"]["questions_per_second"],
            reverse=True,
        )

        print("\n  Results sorted by speed (fastest first):")
        for i, (name, data) in enumerate(sorted_results, 1):
            r = data["results"]
            print(
                f"    {i}. {name}: {r['questions_per_second']:.2f} q/s, "
                f"{r['elapsed_time']:.2f}s, accuracy: {r['accuracy']:.4f}"
            )

        fastest_name, fastest_data = sorted_results[0]
        slowest_name, slowest_data = sorted_results[-1]

        fastest_speed = fastest_data["results"]["questions_per_second"]
        slowest_speed = slowest_data["results"]["questions_per_second"]

        if slowest_speed > 0:
            improvement = ((fastest_speed - slowest_speed) / slowest_speed) * 100
            print(f"\n  Best configuration: {fastest_name}")
            print(f"  Speed improvement vs slowest: {improvement:+.2f}%")
    elif len(valid_results) == 1:
        name, data = list(valid_results.items())[0]
        r = data["results"]
        print(f"\n  Only one successful test: {name}")
        print(f"    Speed: {r['questions_per_second']:.2f} q/s")
        print(f"    Time: {r['elapsed_time']:.2f}s")
        print(f"    Accuracy: {r['accuracy']:.4f}")
    else:
        print("\n  No successful tests to compare")

    print()
    print("=" * 70)

    return results


async def main():
    """Main function"""
    print("=" * 70)
    print("LLM Inference Performance Comparison Tool")
    print("=" * 70)

    config = EvalConfig()

    questions = load_questions(config.QUESTION_FILE)

    if not questions:
        print("\nNo questions loaded. Exiting...")
        return

    print(f"\nTotal questions available: {len(questions)}")
    print(f"Test set size (to evaluate): {len(questions[config.DEV_SET_SIZE:])}")

    subset_size = 10
    print(f"\nNote: For faster testing, we'll use a subset of {subset_size} questions")

    results = await run_comparison(
        questions,
        subset_size=subset_size,
        concurrent_sizes=[10, 20, 30],
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    comparison_file = os.path.join(
        output_dir, f"performance_comparison_{timestamp}.json"
    )

    serializable_results = {}
    for k, v in results.items():
        if v.get("results"):
            serializable_results[k] = {
                "elapsed": v["elapsed"],
                "results": {
                    key: val
                    for key, val in v["results"].items()
                    if key != "detailed_results"
                },
            }
        else:
            serializable_results[k] = v

    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Comparison results saved to {comparison_file}")


if __name__ == "__main__":
    asyncio.run(main())
