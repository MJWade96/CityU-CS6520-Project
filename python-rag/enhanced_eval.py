"""Native enhanced-evaluation entrypoint."""

from __future__ import annotations

import asyncio

from app.rag.evaluation.enhanced_rag_eval import (
    EnhancedEvaluationConfig,
    run_enhanced_evaluation,
)


async def main_async() -> None:
    config = EnhancedEvaluationConfig()
    result = await run_enhanced_evaluation(config)

    print("=" * 60)
    print("Enhanced RAG Evaluation Complete")
    print("=" * 60)
    print(f"Aligned dev set size: {result['dev_set_size']}")
    print(
        f"Test accuracy: {result['test_results']['accuracy']:.4f} "
        f"({result['test_results']['correct']}/{result['test_results']['total_questions']})"
    )
    print(f"JSON results: {result['output_paths']['json']}")
    print(f"Summary: {result['output_paths']['summary']}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


__all__ = ["EnhancedEvaluationConfig", "main", "main_async", "run_enhanced_evaluation"]