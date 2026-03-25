"""
CLI entrypoint for naive RAG evaluation.

The implementation lives in ``app.rag.naive_rag_eval`` so this file stays
small and focused on argument parsing.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.rag.naive_rag_eval import NaiveRAGEvalConfig, run_complete_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run naive RAG evaluation")
    parser.add_argument("--dev-size", type=int, default=300, help="Development set size")
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Test set size (default: evaluate all remaining questions)",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Manual top-k value")
    parser.add_argument(
        "--auto-top-k",
        action="store_true",
        help="Search the configured top-k values on the dev set",
    )
    parser.add_argument("--vector-store", type=Path, help="Override FAISS index path")
    parser.add_argument("--question-file", type=Path, help="Override MedQA file")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    defaults = NaiveRAGEvalConfig()
    config = NaiveRAGEvalConfig(
        dev_size=args.dev_size,
        test_size=args.test_size,
        manual_top_k=None if args.auto_top_k else args.top_k,
        vector_store_path=args.vector_store or defaults.vector_store_path,
        question_file=args.question_file or defaults.question_file,
        output_dir=args.output_dir or defaults.output_dir,
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
