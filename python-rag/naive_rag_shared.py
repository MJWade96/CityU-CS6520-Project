"""
Shared utilities for naive RAG retrieval and generation scripts.

This module centralizes common constants, cache functions, and worker patterns
to avoid code duplication between retrieval and generation scripts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from app.rag.data_paths import EVALUATION_DIR, EVALUATION_RESULTS_DIR, FAISS_INDEX_DIR
from app.rag.eval_shared import ConcurrencyConfig, load_questions, split_questions


SAMPLE_SIZE = 973
TOP_K = 3
DEV_SIZE = 300
QUESTION_FILE = EVALUATION_DIR / "medqa.json"
OUTPUT_DIR = EVALUATION_RESULTS_DIR
VECTOR_STORE_PATH = FAISS_INDEX_DIR
CACHE_DIR = EVALUATION_RESULTS_DIR / "retrieval_cache"


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
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_retrieval_cache(question: str, result: Dict[str, Any]) -> None:
    """Save retrieval result to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(get_question_hash(question))
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def load_sample_questions(
    question_file: Path, dev_size: int, sample_size: int
) -> List[Dict[str, Any]]:
    """Load and split questions, returning test sample."""
    all_questions = load_questions(str(question_file))
    _, test_questions = split_questions(all_questions, dev_size, None)
    return test_questions[:sample_size]


async def run_concurrent_workers(
    items: List[Dict[str, Any]],
    process_item: Callable[[Dict[str, Any]], Any],
    max_concurrent: int,
) -> List[Any]:
    """
    Run concurrent workers using queue-based pattern.

    Args:
        items: List of items to process
        process_item: Async function to process each item
        max_concurrent: Maximum number of concurrent workers

    Returns:
        List of results from processing each item
    """
    results: List[Any] = []
    queue: asyncio.Queue[Dict[str, Any] | None] = asyncio.Queue()
    for item in items:
        await queue.put(item)

    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            try:
                result = await process_item(item)
                async with lock:
                    results.append(result)
            except Exception as e:
                print(f"  Error processing item: {e}")
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(max_concurrent)]
    await queue.join()

    for _ in range(max_concurrent):
        await queue.put(None)
    await asyncio.gather(*workers, return_exceptions=True)

    return results
