"""
Shared utilities for naive RAG retrieval and generation scripts.

This module centralizes common constants, cache functions, and worker patterns
to avoid code duplication between retrieval and generation scripts.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Generic, List, Sequence, Tuple, TypeVar

from app.rag.data.data_paths import (
    EVALUATION_RESULTS_DIR,
    FAISS_INDEX_DIR,
    MEDQA_FILE,
    RETRIEVAL_CACHE_DIR,
)
from app.rag.evaluation.eval_shared import ConcurrencyConfig, load_questions, split_questions
from app.rag.data.json_utils import load_json_safe, save_json_atomic
from app.rag.utils.progress_manager import EvaluationProgressManager


SAMPLE_SIZE = 973
TOP_K = 3
DEV_SIZE = 300
# Re-export canonical evaluation paths so consumers keep one stable import surface.
QUESTION_FILE = MEDQA_FILE
OUTPUT_DIR = EVALUATION_RESULTS_DIR
VECTOR_STORE_PATH = FAISS_INDEX_DIR
CACHE_DIR = RETRIEVAL_CACHE_DIR

ResultT = TypeVar("ResultT")


@dataclass
class WorkerResult(Generic[ResultT]):
    """Shared worker output so callers reuse one queue runner instead of duplicating it."""

    payload: ResultT
    increment_correct: bool = False


def get_question_hash(question: str) -> str:
    """Generate a hash for a question to use as cache key."""
    return hashlib.md5(question.encode("utf-8")).hexdigest()


def get_cache_path(question_hash: str) -> Path:
    """Get the cache file path for a question hash."""
    return CACHE_DIR / f"{question_hash}.json"


def load_cached_retrieval(question: str) -> Dict[str, Any] | None:
    """Load cached retrieval result for a question."""
    cache_path = get_cache_path(get_question_hash(question))
    if cache_path.exists():
        return load_json_safe(cache_path)
    return None


def save_retrieval_cache(question: str, result: Dict[str, Any]) -> None:
    """Save retrieval result to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(get_question_hash(question))
    save_json_atomic(cache_path, result)


def load_sample_questions(
    question_file: Path, dev_size: int, sample_size: int
) -> List[Dict[str, Any]]:
    """Load and split questions, returning test sample."""
    all_questions = load_questions(str(question_file))
    _, test_questions = split_questions(all_questions, dev_size, None)
    return test_questions[:sample_size]


def write_live_sample_progress(
    progress_mgr: EvaluationProgressManager,
    artifact_paths: Dict[str, Path],
    live_config: Dict[str, Any],
    *,
    console_run_name: str,
    artifact_run_name: str,
    evaluation_type: str,
    total_questions: int,
    processed_questions: int,
    correct_count: int,
    elapsed: float,
    results: List[Dict[str, Any]],
    top_k: int,
) -> None:
    """Keep sample progress output and live artifacts aligned across naive-RAG scripts."""
    progress_mgr.print_progress(
        run_name=console_run_name,
        dataset_name="Sample",
        processed_questions=processed_questions,
        total_questions=total_questions,
        correct_count=correct_count,
        elapsed_time=elapsed,
    )

    stage_result = progress_mgr.build_stage_result(
        dataset_name="Sample",
        total_questions=total_questions,
        processed_questions=processed_questions,
        correct_count=correct_count,
        elapsed_time=elapsed,
        detailed_results=results,
        top_k=top_k,
    )
    progress_mgr.write_live_results(
        artifact_paths=artifact_paths,
        run_name=artifact_run_name,
        evaluation_type=evaluation_type,
        config=live_config,
        stage_result=stage_result,
        status="running",
    )


async def run_tracked_workers(
    items: Sequence[Dict[str, Any]],
    process_item: Callable[[Dict[str, Any]], Awaitable[WorkerResult[ResultT]]],
    max_concurrent: int,
    on_progress: Callable[[List[ResultT], int, int, float], None] | None = None,
    on_error: Callable[[Dict[str, Any], Exception], WorkerResult[ResultT] | None] | None = None,
) -> Tuple[List[ResultT], int]:
    """Run one shared worker loop so callers only provide item-specific logic."""
    results: List[ResultT] = []
    correct_count = 0
    queue: asyncio.Queue[Dict[str, Any] | None] = asyncio.Queue()
    for item in items:
        await queue.put(item)

    lock = asyncio.Lock()
    start_time = time.time()

    async def worker() -> None:
        nonlocal correct_count
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            try:
                try:
                    outcome = await process_item(item)
                except Exception as exc:
                    if on_error is None:
                        print(f"  Error processing item: {exc}")
                        continue
                    outcome = on_error(item, exc)
                    if outcome is None:
                        continue

                async with lock:
                    results.append(outcome.payload)
                    if outcome.increment_correct:
                        correct_count += 1
                    progress_results = list(results)
                    processed = len(progress_results)
                    progress_correct = correct_count

                if on_progress is not None:
                    on_progress(
                        progress_results,
                        processed,
                        progress_correct,
                        time.time() - start_time,
                    )
            except Exception as exc:
                print(f"  Worker encountered unexpected error: {exc}")
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(max_concurrent)]
    await queue.join()

    for _ in range(max_concurrent):
        await queue.put(None)
    await asyncio.gather(*workers, return_exceptions=True)

    return results, correct_count
