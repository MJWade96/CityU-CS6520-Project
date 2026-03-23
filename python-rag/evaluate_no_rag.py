"""
Evaluate LLM accuracy WITHOUT RAG (direct LLM inference)

This script evaluates the baseline performance of the LLM without any
retrieval augmentation - using only the model's internal knowledge.

Usage:
    python evaluate_no_rag.py
"""

import os
import sys
import json
import time
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

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

    MAX_CONCURRENT = 20


NO_RAG_PROMPT = """You are a medical expert assistant. Answer the following question based on your medical knowledge.

Question: {question}

Options:
{options}

Please think step by step and then provide your answer in the following format:
Answer: [A/B/C/D/E]

Your response:"""


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


async def evaluate_without_rag(
    client: AsyncOpenAI,
    questions: List[Dict],
    config: EvalConfig,
) -> Dict[str, Any]:
    """
    Evaluate LLM without RAG (direct inference) using async parallel calls.

    Flow:
    1. Send questions in parallel to LLM (no retrieval)
    2. Extract answer from responses
    3. Compare with correct answers

    Returns:
        Evaluation results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating WITHOUT RAG - Full Dataset (Async)")
    print(f"{'=' * 60}")

    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT)

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


async def main():
    """Main evaluation function"""
    print("=" * 60)
    print("Medical RAG System - Baseline Evaluation (WITHOUT RAG)")
    print("=" * 60)

    config = EvalConfig()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    questions = load_questions(config.QUESTION_FILE)

    if not questions:
        print("\nNo questions loaded. Exiting...")
        return

    print(f"\nInitializing Async LLM Client (No RAG)...")
    print(f"  Provider: {config.LLM_PROVIDER}")
    print(f"  Model: {config.LLM_MODEL}")
    print(f"  Temperature: {config.LLM_TEMPERATURE}")
    print(f"  Base URL: {config.LLM_BASE_URL}")
    print(f"  Max Concurrent: {config.MAX_CONCURRENT}")

    client = AsyncOpenAI(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
    )
    print("[OK] Async LLM Client initialized")

    no_rag_results = await evaluate_without_rag(
        client,
        questions,
        config,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    results_file = os.path.join(config.OUTPUT_DIR, f"no_rag_eval_{timestamp}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "llm_provider": config.LLM_PROVIDER,
                    "llm_model": config.LLM_MODEL,
                    "evaluation_type": "NO_RAG (Direct LLM Inference - Async)",
                    "max_concurrent": config.MAX_CONCURRENT,
                },
                "evaluation_results": no_rag_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[OK] Results saved to {results_file}")

    summary_file = os.path.join(config.OUTPUT_DIR, f"no_rag_summary_{timestamp}.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Medical RAG System - Baseline Evaluation (WITHOUT RAG)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL}\n")
        f.write("Evaluation Type: Direct LLM Inference (No Retrieval - Async)\n")
        f.write(f"Max Concurrent: {config.MAX_CONCURRENT}\n\n")

        f.write("Results:\n")
        f.write(f"  Total Questions: {no_rag_results['total_questions']}\n")
        f.write(f"  Correct Answers: {no_rag_results['correct']}\n")
        f.write(f"  Accuracy: {no_rag_results['accuracy']:.4f}\n")
        f.write(f"  Time: {no_rag_results['elapsed_time']:.1f}s\n")
        f.write(f"  Speed: {no_rag_results['questions_per_second']:.2f} q/s\n\n")

    print(f"[OK] Summary saved to {summary_file}")

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n📊 Baseline Results (No RAG):")
    print(f"  Total Questions: {no_rag_results['total_questions']}")
    print(f"  Correct Answers: {no_rag_results['correct']}")
    print(
        f"\n🎯 Accuracy: {no_rag_results['accuracy']:.4f} "
        f"({no_rag_results['correct']}/{no_rag_results['total_questions']})"
    )
    print(
        f"\n⏱️  Evaluation Time: {no_rag_results['elapsed_time']:.1f}s "
        f"({no_rag_results['questions_per_second']:.2f} questions/second)"
    )

    print(f"\n{'=' * 60}")
    print("This is the BASELINE performance without RAG.")
    print("Compare with RAG-enhanced results to measure improvement.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
