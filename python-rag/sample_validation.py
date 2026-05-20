"""CLI entrypoint for sample validation.

The implementation lives in ``app.rag.evaluation.sample_validation_eval`` so
this file stays small and focused on configuration.
"""

from __future__ import annotations

import asyncio

from app.rag.evaluation.sample_validation_eval import SampleEvalConfig, run_sample_comparison


async def main() -> None:
    config = SampleEvalConfig()
    await run_sample_comparison(config)


if __name__ == "__main__":
    asyncio.run(main())