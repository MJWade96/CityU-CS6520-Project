"""
CLI entrypoint for baseline evaluation without retrieval.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.rag.no_rag_eval import NoRAGEvalConfig, run_no_rag_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline evaluation without RAG")
    parser.add_argument("--dev-size", type=int, default=300, help="Development set size")
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Test set size (default: evaluate all remaining questions)",
    )
    parser.add_argument("--question-file", type=Path, help="Override MedQA file")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    defaults = NoRAGEvalConfig()
    config = NoRAGEvalConfig(
        dev_size=args.dev_size,
        test_size=args.test_size,
        question_file=args.question_file or defaults.question_file,
        output_dir=args.output_dir or defaults.output_dir,
    )
    result = await run_no_rag_evaluation(config)

    print("=" * 60)
    print("Baseline Evaluation Complete")
    print("=" * 60)
    print(
        f"Accuracy: {result['results']['accuracy']:.4f} "
        f"({result['results']['correct']}/{result['results']['total_questions']})"
    )
    print(f"JSON results: {result['output_paths']['json']}")
    print(f"Summary: {result['output_paths']['summary']}")


if __name__ == "__main__":
    asyncio.run(main())
