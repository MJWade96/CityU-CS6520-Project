"""Native enhanced-evaluation entrypoint."""

from __future__ import annotations

import asyncio

from app.rag.evaluation.enhanced_rag_eval import (
    EnhancedEvaluationConfig,
    run_enhanced_evaluation,
)


async def main_async() -> None:
    config = EnhancedEvaluationConfig()
    await run_enhanced_evaluation(config)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


__all__ = ["EnhancedEvaluationConfig", "main", "main_async", "run_enhanced_evaluation"]
