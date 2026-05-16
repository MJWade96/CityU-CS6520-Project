"""
Naive RAG retrieval script - performs retrieval and caches results.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from app.rag.data.json_utils import save_json_atomic
from app.rag.evaluation.naive_rag_eval import load_vector_store
from app.rag.utils.progress_manager import EvaluationProgressManager
from naive_rag_shared import (
    CACHE_DIR,
    ConcurrencyConfig,
    load_cached_retrieval,
    load_sample_questions,
    OUTPUT_DIR,
    QUESTION_FILE,
    run_tracked_workers,
    SAMPLE_SIZE,
    save_retrieval_cache,
    TOP_K,
    VECTOR_STORE_PATH,
    DEV_SIZE,
    WorkerResult,
)


@dataclass
class RetrievalConfig:
    sample_size: int = SAMPLE_SIZE
    top_k: int = TOP_K
    dev_size: int = DEV_SIZE
    question_file: Path = QUESTION_FILE
    output_dir: Path = OUTPUT_DIR
    vector_store_path: Path = VECTOR_STORE_PATH
    cache_dir: Path = CACHE_DIR
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


async def run_retrieval(config: RetrievalConfig) -> Dict[str, Any]:
    """Run retrieval for all questions and save cached results."""
    sample_questions = load_sample_questions(
        config.question_file, config.dev_size, config.sample_size
    )

    progress_mgr = EvaluationProgressManager(output_dir=str(config.output_dir))
    artifact_paths = progress_mgr.create_run_artifacts("naive_rag_retrieval")

    print("=" * 60 + "\nNaive RAG Retrieval\n" + "=" * 60)
    print(f"Sample size: {config.sample_size}\nTop-k: {config.top_k}\n")

    print("Loading vector store...")
    try:
        vectorstore = load_vector_store(config.vector_store_path)
        print("  Vector store loaded successfully")
    except Exception as e:
        print(f"  ERROR: Failed to load vector store: {e}")
        return {"error": str(e)}

    start_time = time.time()
    cached_count = 0

    async def process_item(item: Dict[str, Any]) -> WorkerResult[Dict[str, Any]]:
        nonlocal cached_count
        question = item.get("question", "")
        cached = load_cached_retrieval(question)

        if cached:
            cached_count += 1
            return WorkerResult(payload=cached)

        search_results = await asyncio.to_thread(
            vectorstore.similarity_search_with_score, question, config.top_k
        )
        docs = [doc for doc, _ in search_results]
        scores = [float(score) for _, score in search_results]
        contexts = [doc.page_content for doc in docs]

        result = {
            "question": question,
            "options": item.get("options", []),
            "answer_index": item.get("answer_index", -1),
            "retrieved_docs": len(docs),
            "scores": scores,
            "contexts": contexts,
        }
        save_retrieval_cache(question, result)
        return WorkerResult(payload=result)

    def handle_progress(
        _results: List[Dict[str, Any]], processed: int, _correct_count: int, elapsed: float
    ) -> None:
        progress_mgr.print_progress(
            run_name="RETRIEVAL",
            dataset_name="Sample",
            processed_questions=processed,
            total_questions=len(sample_questions),
            correct_count=cached_count,
            elapsed_time=elapsed,
        )

    print("Running retrieval...")
    results, _correct_count = await run_tracked_workers(
        items=sample_questions,
        process_item=process_item,
        max_concurrent=config.concurrency.max_concurrent,
        on_progress=handle_progress,
    )

    elapsed = time.time() - start_time
    output = {
        "total_questions": len(sample_questions),
        "processed_questions": len(results),
        "cached_count": cached_count,
        "elapsed_time": elapsed,
        "retrieval_results": results,
    }

    print(f"  Cached: {cached_count}/{len(sample_questions)}")
    print(f"  Time: {elapsed:.1f}s\n")

    output_file = config.output_dir / "retrieval_results.json"
    save_json_atomic(output_file, output)
    print(f"Results saved to: {output_file}")

    return output


async def main() -> None:
    config = RetrievalConfig()
    await run_retrieval(config)


if __name__ == "__main__":
    asyncio.run(main())
