"""
Small-sample validation script to compare naive-RAG vs no-RAG before full evaluation.

This script runs a quick comparison on a small subset of questions to verify
that naive-RAG provides improvement over the no-RAG baseline before committing
to the full 973-question evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.rag.data_paths import EVALUATION_DIR, EVALUATION_RESULTS_DIR, FAISS_INDEX_DIR
from app.rag.embeddings import get_langchain_embeddings
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
)
from app.rag.vector_store import MedicalVectorStore


@dataclass
class SampleEvalConfig:
    sample_size: int = 50
    question_file: Path = EVALUATION_DIR / "medqa.json"
    output_dir: Path = EVALUATION_RESULTS_DIR
    vector_store_path: Path = FAISS_INDEX_DIR
    top_k: int = 3
    llm: EvaluationLLMConfig = field(default_factory=EvaluationLLMConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


def load_vector_store_for_eval(index_path: Path) -> MedicalVectorStore:
    """Load the persisted FAISS store with BGE-M3 embeddings."""
    embeddings = get_langchain_embeddings(
        model_type="bge-m3",
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = MedicalVectorStore(
        embedding_model=embeddings,
        store_type="faiss",
        persist_directory=str(index_path),
    )
    vectorstore.load(str(index_path))
    return vectorstore


async def evaluate_no_rag_sample(
    questions: List[Dict[str, Any]],
    config: SampleEvalConfig,
) -> Dict[str, Any]:
    """Evaluate without retrieval on a small sample."""
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
            "question": item["question"][:100] + "...",
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer == correct_answer,
        }

    start_time = time.time()
    results = await asyncio.gather(*(evaluate_item(item) for item in questions))
    elapsed = time.time() - start_time

    correct = sum(1 for r in results if r["is_correct"])
    return {
        "mode": "NO_RAG",
        "total_questions": len(questions),
        "correct": correct,
        "accuracy": correct / len(questions) if questions else 0.0,
        "elapsed_time": elapsed,
        "questions_per_second": len(questions) / elapsed if elapsed > 0 else 0.0,
        "detailed_results": results,
    }


async def evaluate_naive_rag_sample(
    questions: List[Dict[str, Any]],
    vectorstore: MedicalVectorStore,
    config: SampleEvalConfig,
) -> Dict[str, Any]:
    """Evaluate with naive RAG on a small sample."""
    client = create_async_client(config.llm)
    semaphore = asyncio.Semaphore(config.concurrency.max_concurrent)
    rate_limiter = RateLimiter(
        requests_per_second=config.concurrency.requests_per_second,
        burst=config.concurrency.max_concurrent,
    )

    async def evaluate_item(item: Dict[str, Any]) -> Dict[str, Any]:
        search_results = await asyncio.to_thread(
            vectorstore.similarity_search_with_score,
            item["question"],
            config.top_k,
        )
        docs = [doc for doc, _ in search_results]
        contexts = [doc.page_content for doc in docs]
        prompt = build_medical_eval_prompt(
            question=item["question"],
            options=item.get("options", []),
            context="\n\n".join(f"[{index + 1}] {context[:200]}..." for index, context in enumerate(contexts)),
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
        return {
            "question": item["question"][:100] + "...",
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer == correct_answer,
            "retrieved_docs": len(docs),
        }

    start_time = time.time()
    results = await asyncio.gather(*(evaluate_item(item) for item in questions))
    elapsed = time.time() - start_time

    correct = sum(1 for r in results if r["is_correct"])
    return {
        "mode": "NAIVE_RAG",
        "top_k": config.top_k,
        "total_questions": len(questions),
        "correct": correct,
        "accuracy": correct / len(questions) if questions else 0.0,
        "elapsed_time": elapsed,
        "questions_per_second": len(questions) / elapsed if elapsed > 0 else 0.0,
        "detailed_results": results,
    }


async def run_sample_comparison(config: SampleEvalConfig) -> Dict[str, Any]:
    """Run comparison between no-RAG and naive-RAG on a small sample."""
    all_questions = load_questions(str(config.question_file))
    sample_questions = all_questions[: config.sample_size]

    print("=" * 60)
    print("Small-Sample Validation: NO_RAG vs NAIVE_RAG")
    print("=" * 60)
    print(f"Sample size: {config.sample_size}")
    print(f"Top-k for RAG: {config.top_k}")
    print(f"LLM model: {config.llm.model}")
    print()

    print("Running NO_RAG evaluation...")
    no_rag_results = await evaluate_no_rag_sample(sample_questions, config)
    print(f"  Accuracy: {no_rag_results['accuracy']:.4f} ({no_rag_results['correct']}/{no_rag_results['total_questions']})")
    print(f"  Time: {no_rag_results['elapsed_time']:.1f}s")
    print()

    print("Loading vector store for NAIVE_RAG...")
    try:
        vectorstore = load_vector_store_for_eval(config.vector_store_path)
        print("  Vector store loaded successfully")
    except Exception as e:
        print(f"  ERROR: Failed to load vector store: {e}")
        print("  Skipping NAIVE_RAG evaluation")
        return {
            "error": str(e),
            "no_rag_results": no_rag_results,
            "naive_rag_results": None,
        }

    print("Running NAIVE_RAG evaluation...")
    naive_rag_results = await evaluate_naive_rag_sample(sample_questions, vectorstore, config)
    print(f"  Accuracy: {naive_rag_results['accuracy']:.4f} ({naive_rag_results['correct']}/{naive_rag_results['total_questions']})")
    print(f"  Time: {naive_rag_results['elapsed_time']:.1f}s")
    print()

    improvement = naive_rag_results["accuracy"] - no_rag_results["accuracy"]
    improvement_pct = (improvement / no_rag_results["accuracy"] * 100) if no_rag_results["accuracy"] > 0 else 0

    print("=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"NO_RAG accuracy:    {no_rag_results['accuracy']:.4f} ({no_rag_results['correct']}/{no_rag_results['total_questions']})")
    print(f"NAIVE_RAG accuracy: {naive_rag_results['accuracy']:.4f} ({naive_rag_results['correct']}/{naive_rag_results['total_questions']})")
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
        print("Recommendation: Investigate retrieval relevance and context integration.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = config.output_dir / f"sample_comparison_{timestamp}.json"
    comparison_result = {
        "config": {
            "sample_size": config.sample_size,
            "top_k": config.top_k,
            "llm_model": config.llm.model,
            "vector_store_path": str(config.vector_store_path),
        },
        "no_rag_results": no_rag_results,
        "naive_rag_results": naive_rag_results,
        "comparison": {
            "improvement": improvement,
            "improvement_percent": improvement_pct,
            "naive_rag_better": improvement > 0,
        },
    }
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(comparison_result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    return comparison_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small-sample validation for RAG comparison")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of questions to evaluate")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k documents to retrieve for RAG")
    parser.add_argument("--question-file", type=Path, help="Override MedQA file")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--vector-store", type=Path, help="Override vector store path")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    defaults = SampleEvalConfig()
    config = SampleEvalConfig(
        sample_size=args.sample_size,
        top_k=args.top_k,
        question_file=args.question_file or defaults.question_file,
        output_dir=args.output_dir or defaults.output_dir,
        vector_store_path=args.vector_store or defaults.vector_store_path,
    )
    await run_sample_comparison(config)


if __name__ == "__main__":
    asyncio.run(main())
