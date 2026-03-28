"""
CLI entrypoint for baseline evaluation without retrieval.
"""

from __future__ import annotations

import asyncio

from app.rag.no_rag_eval import NoRAGEvalConfig, run_no_rag_evaluation


DEV_SIZE = 300
TEST_SIZE = None
QUESTION_FILE = None
OUTPUT_DIR = None


async def main() -> None:
    defaults = NoRAGEvalConfig()
    config = NoRAGEvalConfig(
        dev_size=DEV_SIZE,
        test_size=TEST_SIZE,
        question_file=QUESTION_FILE or defaults.question_file,
        output_dir=OUTPUT_DIR or defaults.output_dir,
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
