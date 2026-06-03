"""Tests for evaluation pipeline concurrency scheduling."""

from __future__ import annotations

import asyncio


def test_pipeline_refills_slot_before_slow_prefix_commits() -> None:
    from app.rag.evaluation.eval_shared import iter_pipeline_in_order

    async def run_pipeline() -> tuple[list[int], list[int]]:
        started: list[int] = []
        yielded: list[int] = []
        release_first = asyncio.Event()
        third_started = asyncio.Event()

        async def worker(index: int, item: int) -> int:
            started.append(index)
            if index == 0:
                await release_first.wait()
            if index == 2:
                third_started.set()
            return item

        async def collect() -> None:
            async for index, _item, result in iter_pipeline_in_order(
                [0, 1, 2],
                max_concurrent=2,
                worker=worker,
            ):
                yielded.append(index)
                assert result == index

        collector = asyncio.create_task(collect())
        await asyncio.wait_for(third_started.wait(), timeout=1.0)
        assert started == [0, 1, 2]
        assert yielded == []

        release_first.set()
        await collector
        return started, yielded

    started, yielded = asyncio.run(run_pipeline())

    assert started == [0, 1, 2]
    assert yielded == [0, 1, 2]
