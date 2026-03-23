"""
Evaluate LLM accuracy WITHOUT RAG (direct LLM inference) - Async Optimized Version

This script evaluates the baseline performance of the LLM without any
retrieval augmentation - using only the model's internal knowledge.

Optimizations:
- Async/await for concurrent API calls
- Batch processing with semaphore-based concurrency control
- Connection pooling for better performance

Usage:
    python evaluate_no_rag_async.py
"""

import os
import sys
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import threading

sys.path.insert(0, str(Path(__file__).parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


@dataclass
class RateLimitConfig:
    TPM: int = 100000
    RPM: int = 1000
    TPS: int = 167
    BURST_SIZE: int = 50


class EvalConfig:
    SCRIPT_DIR = Path(__file__).parent
    DEV_SET_SIZE = 300
    LLM_PROVIDER = "deepseek"
    LLM_MODEL = "2656053fa69c4c2d89c5a691d9d737c3"
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 512
    LLM_BASE_URL = "https://wishub-x6.ctyun.cn/v1"
    LLM_API_KEY = "6fcecb364d0647d2883e7f1d3f19d5b9"
    RATE_LIMIT = RateLimitConfig()
    BATCH_SIZE = 20
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    TIMEOUT = 60
    QUESTION_FILE = str(SCRIPT_DIR / "data" / "evaluation" / "medqa.json")
    OUTPUT_DIR = str(SCRIPT_DIR / "results" / "evaluation")


NO_RAG_PROMPT = """You are a medical expert assistant. Answer the following medical question based on your knowledge.

Question: {question}

Options:
{options}

Please analyze this question step by step:
1. Identify the key medical concepts in the question
2. Consider what knowledge is needed to answer correctly
3. Evaluate each option systematically
4. Select the best answer

Provide your answer in the following format:
Answer: [A/B/C/D/E]

Your response:"""


class AsyncRateLimiter:
    def __init__(self, rpm: int = 1000, tps: int = 167):
        self.rpm = rpm
        self.tps = tps
        self.lock = asyncio.Lock()
        self.request_timestamps = []
        self.token_count = 0
        self.last_reset_time = time.time()

    def _reset_if_needed(self):
        current_time = time.time()
        if current_time - self.last_reset_time >= 60:
            self.request_timestamps = []
            self.token_count = 0
            self.last_reset_time = current_time

    async def acquire(self, token_cost: int = 100):
        async with self.lock:
            self._reset_if_needed()

            while True:
                current_time = time.time()

                if current_time - self.last_reset_time >= 60:
                    self.request_timestamps = []
                    self.token_count = 0
                    self.last_reset_time = current_time
                    continue

                valid_requests = [
                    t for t in self.request_timestamps if current_time - t < 60
                ]
                self.request_timestamps = valid_requests

                if (
                    len(self.request_timestamps) < self.rpm
                    and self.token_count < self.tps
                ):
                    self.request_timestamps.append(current_time)
                    self.token_count += token_cost
                    return

                if self.request_timestamps:
                    oldest = min(self.request_timestamps)
                    sleep_time = max(0.01, (oldest + 60) - current_time)
                    await asyncio.sleep(min(sleep_time, 0.1))


class AsyncDirectLLMGenerator:
    def __init__(
        self,
        provider: str = "deepseek",
        model: str = "2656053fa69c4c2d89c5a691d9d737c3",
        temperature: float = 0.1,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        rate_limiter: Optional[AsyncRateLimiter] = None,
        max_concurrent: int = 10,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.api_key = api_key or self._get_api_key(provider)
        if not self.api_key:
            raise ValueError(
                f"API key not found for {provider}. "
                f"Please set {provider.upper()}_API_KEY environment variable."
            )

        self.base_url = base_url or self._get_base_url(provider)

        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=EvalConfig.TIMEOUT,
        )

        self.rate_limiter = rate_limiter or AsyncRateLimiter(
            rpm=EvalConfig.RATE_LIMIT.RPM,
            tps=EvalConfig.RATE_LIMIT.TPS,
        )
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def _get_api_key(self, provider: str) -> Optional[str]:
        provider_keys = {
            "openai": "OPENAI_API_KEY",
            "zhipu": "ZHIPU_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "moonshot": "MOONSHOT_API_KEY",
        }
        env_var = provider_keys.get(provider.lower(), "OPENAI_API_KEY")
        return os.getenv(env_var)

    def _get_base_url(self, provider: str) -> str:
        base_urls = {
            "openai": "https://api.openai.com/v1",
            "zhipu": "https://open.bigmodel.cn/api/paas/v4",
            "deepseek": "https://api.deepseek.com/v1",
            "moonshot": "https://api.moonshot.cn/v1",
        }
        return base_urls.get(provider.lower(), "https://api.deepseek.com/v1")

    def _estimate_token_cost(self, text: str) -> int:
        return len(text) // 4

    def _format_prompt(self, question: str, options: Optional[List[str]]) -> str:
        if options:
            options_text = "\n".join(
                [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
            )
        else:
            options_text = (
                "A. Not provided\nB. Not provided\nC. Not provided\nD. Not provided"
            )

        return NO_RAG_PROMPT.format(
            question=question,
            options=options_text,
        )

    async def generate_async(
        self,
        question: str,
        options: Optional[List[str]] = None,
        max_retries: int = 3,
    ) -> str:
        prompt = self._format_prompt(question, options)
        token_cost = self._estimate_token_cost(prompt) + self.max_tokens

        for attempt in range(max_retries):
            try:
                await self.rate_limiter.acquire(token_cost)

                async with self.semaphore:
                    response = await self.llm.agenerate([prompt])
                    return response.generations[0][0].text
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(EvalConfig.RETRY_DELAY * (attempt + 1))
                else:
                    return f"Error generating answer: {str(e)}"

        return f"Error generating answer after {max_retries} attempts"

    def extract_answer(self, response: str) -> Optional[str]:
        import re

        if not response or not isinstance(response, str):
            return None

        lines = response.strip().split("\n")
        for line in reversed(lines[-5:]):
            answer_match = re.search(r"^\s*Answer:\s*([A-E])\s*$", line, re.IGNORECASE)
            if answer_match:
                return answer_match.group(1).upper()

        answer_match = re.search(r"Answer:\s*([A-E])\b", response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()

        bracket_match = re.search(r"[\[\(]([A-E])[\]\)]", response)
        if bracket_match:
            return bracket_match.group(1).upper()

        last_line = lines[-1] if lines else response
        conclusion_patterns = [
            r"(?:answer|choice|option|is)\s*[is:]?\s*([A-E])\b",
        ]
        for pattern in conclusion_patterns:
            match = re.search(pattern, last_line, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        last_line_clean = last_line.strip().strip(".").strip(":")
        if len(last_line_clean) == 1 and last_line_clean.upper() in "ABCDE":
            return last_line_clean.upper()

        return None


def load_questions(question_file: str) -> List[Dict]:
    print(f"Loading questions from {question_file}...")

    if not os.path.exists(question_file):
        print(f"ERROR: File not found: {question_file}")
        return []

    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"[OK] Loaded {len(questions)} questions")
    return questions


async def evaluate_without_rag_async(
    llm_generator: AsyncDirectLLMGenerator,
    questions: List[Dict],
    dataset_name: str = "Dataset",
    batch_size: int = 20,
    output_dir: str = None,
) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"Evaluating WITHOUT RAG (Async) - {dataset_name}")
    print(f"{'=' * 60}")
    print(f"  Async processing: Enabled (batch_size={batch_size})")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_results_file = os.path.join(output_dir, f"no_rag_async_temp_{timestamp}.json")
        print(f"  Temp results file: {temp_results_file}")
    else:
        temp_results_file = None

    start_time = time.time()
    results = []
    correct = 0
    total = 0
    processed = 0

    async def process_single_question(q: Dict, idx: int) -> Dict:
        nonlocal processed, correct, total

        question_text = q.get("question", "")
        options = q.get("options", [])
        correct_answer = q.get("answer", "")
        answer_index = q.get("answer_index", -1)

        if answer_index >= 0:
            correct_answer_letter = chr(65 + answer_index)
        else:
            correct_answer_letter = correct_answer

        response = await llm_generator.generate_async(question_text, options)
        predicted_answer = llm_generator.extract_answer(response)
        is_correct = predicted_answer == correct_answer_letter.upper()

        result = {
            "question": question_text,
            "options": options,
            "correct_answer": correct_answer_letter,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "response": response,
        }

        async with asyncio.Lock():
            processed += 1
            if is_correct:
                correct += 1
            total += 1
            results.append(result)

            if processed % 10 == 0 or processed == len(questions):
                elapsed = time.time() - start_time
                qps = processed / elapsed if elapsed > 0 else 0
                current_acc = correct / total if total > 0 else 0
                print(
                    f"  Question {processed}/{len(questions)} | "
                    f"Accuracy: {current_acc:.4f} | "
                    f"Speed: {qps:.2f} q/s"
                )

                if temp_results_file:
                    with open(temp_results_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "dataset_name": dataset_name,
                                "total_questions": len(questions),
                                "processed": processed,
                                "current_accuracy": current_acc,
                                "elapsed_time": elapsed,
                                "results": results,
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )

        return result

    tasks = [
        process_single_question(q, i) for i, q in enumerate(questions)
    ]

    await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0

    final_results = {
        "dataset_name": dataset_name,
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_time": elapsed,
        "questions_per_second": total / elapsed if elapsed > 0 else 0,
        "detailed_results": results,
    }

    if temp_results_file:
        with open(temp_results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_name": dataset_name,
                    "total_questions": total,
                    "processed": total,
                    "current_accuracy": accuracy,
                    "elapsed_time": elapsed,
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    return final_results


def main():
    print("=" * 60)
    print("Medical RAG System - Baseline Evaluation (WITHOUT RAG) [ASYNC]")
    print("=" * 60)

    config = EvalConfig()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    questions = load_questions(config.QUESTION_FILE)

    if not questions:
        print("\nNo questions loaded. Exiting...")
        return

    dev_set = questions[: config.DEV_SET_SIZE]
    test_set = questions[config.DEV_SET_SIZE :]

    print(f"\nDataset Split:")
    print(f"  Development set: {len(dev_set)} questions")
    print(f"  Test set: {len(test_set)} questions")

    print(f"\nInitializing Async LLM Generator (No RAG)...")
    print(f"  Provider: {config.LLM_PROVIDER}")
    print(f"  Model: {config.LLM_MODEL}")
    print(f"  Temperature: {config.LLM_TEMPERATURE}")
    print(f"  Base URL: {config.LLM_BASE_URL}")
    print(f"  Max Concurrent: {config.BATCH_SIZE}")

    try:
        rate_limiter = AsyncRateLimiter(
            rpm=EvalConfig.RATE_LIMIT.RPM,
            tps=EvalConfig.RATE_LIMIT.TPS,
        )

        llm_generator = AsyncDirectLLMGenerator(
            provider=config.LLM_PROVIDER,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            rate_limiter=rate_limiter,
            max_concurrent=config.BATCH_SIZE,
        )
        print("[OK] Async LLM Generator initialized")
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nPlease check your API configuration")
        return

    no_rag_results = asyncio.run(
        evaluate_without_rag_async(
            llm_generator,
            test_set,
            dataset_name="Test Set (No RAG Async)",
            batch_size=config.BATCH_SIZE,
            output_dir=config.OUTPUT_DIR,
        )
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    results_file = os.path.join(config.OUTPUT_DIR, f"no_rag_async_eval_{timestamp}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "llm_provider": config.LLM_PROVIDER,
                    "llm_model": config.LLM_MODEL,
                    "evaluation_type": "NO_RAG (Direct LLM Inference) [ASYNC]",
                    "max_concurrent": config.BATCH_SIZE,
                },
                "evaluation_results": no_rag_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[OK] Results saved to {results_file}")
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total Questions: {no_rag_results['total_questions']}")
    print(f"  Correct: {no_rag_results['correct']}")
    print(f"  Accuracy: {no_rag_results['accuracy']:.4f}")
    print(f"  Elapsed Time: {no_rag_results['elapsed_time']:.2f} seconds")
    print(f"  Speed: {no_rag_results['questions_per_second']:.2f} questions/second")


if __name__ == "__main__":
    main()
