"""
CLI entrypoint for naive RAG evaluation.

The implementation lives in ``app.rag.naive_rag_eval`` so this file stays
small and focused on configuration.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.rag.naive_rag_eval import NaiveRAGEvalConfig, run_complete_evaluation
from app.rag.data_paths import EVALUATION_DIR, EVALUATION_RESULTS_DIR, FAISS_INDEX_DIR


DEV_SIZE = 300
TEST_SIZE = None
TOP_K = 3
AUTO_TOP_K = False
VECTOR_STORE_PATH = FAISS_INDEX_DIR
QUESTION_FILE = EVALUATION_DIR / "medqa.json"
OUTPUT_DIR = EVALUATION_RESULTS_DIR


async def main() -> None:
    defaults = NaiveRAGEvalConfig()
    config = NaiveRAGEvalConfig(
        dev_size=DEV_SIZE,
        test_size=TEST_SIZE,
        manual_top_k=None if AUTO_TOP_K else TOP_K,
        vector_store_path=VECTOR_STORE_PATH,
        question_file=QUESTION_FILE,
        output_dir=OUTPUT_DIR,
    )
    result = await run_complete_evaluation(config)

    print("=" * 60)
    print("Naive RAG Evaluation Complete")
    print("=" * 60)
    print(f"Best top-k: {result['best_k']}")
    print(
        f"Test accuracy: {result['test_results']['accuracy']:.4f} "
        f"({result['test_results']['correct']}/{result['test_results']['total_questions']})"
    )
    print(f"Recall@k: {result['recall_scores']}")
    print(f"JSON results: {result['output_paths']['json']}")
    print(f"Summary: {result['output_paths']['summary']}")


if __name__ == "__main__":
    asyncio.run(main())
