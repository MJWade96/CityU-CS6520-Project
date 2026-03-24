"""
Shared helpers for baseline and naive RAG evaluation scripts.

Centralizing these utilities keeps prompt formatting, answer extraction,
dataset splitting, and rate limiting consistent across evaluation modes.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI

from .data_paths import EVALUATION_DIR


DEFAULT_BASE_URL = "https://wishub-x6.ctyun.cn/v1"
DEFAULT_MODEL = "8606056bfe0c49448d92587452d1f2fc"
DEFAULT_PROVIDER = "Qwen3-4B"
DEFAULT_API_KEY = "4dbe3bec3ee548d28b649b324e741939"


@dataclass
class EvaluationLLMConfig:
    provider: str = os.getenv("RAG_LLM_PROVIDER", DEFAULT_PROVIDER)
    model: str = os.getenv("RAG_LLM_MODEL", DEFAULT_MODEL)
    temperature: float = float(os.getenv("RAG_LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("RAG_LLM_MAX_TOKENS", "512"))
    base_url: str = os.getenv("RAG_LLM_BASE_URL", DEFAULT_BASE_URL)
    api_key: str = os.getenv("RAG_LLM_API_KEY", DEFAULT_API_KEY)


@dataclass
class ConcurrencyConfig:
    rpm_limit: int = int(os.getenv("RAG_EVAL_RPM_LIMIT", "1000"))
    max_concurrent: int = int(os.getenv("RAG_EVAL_MAX_CONCURRENT", "10"))

    @property
    def requests_per_second(self) -> float:
        return self.rpm_limit / 60 * 0.9


def load_questions(question_file: Optional[str] = None) -> List[Dict]:
    """Load MedQA questions from JSON."""
    question_path = Path(question_file or EVALUATION_DIR / "medqa.json")
    if not question_path.exists():
        raise FileNotFoundError(f"Question file not found: {question_path}")

    with question_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_questions(
    questions: Sequence[Dict],
    dev_size: int,
    test_size: Optional[int],
) -> Tuple[List[Dict], List[Dict]]:
    """Split questions into dev and test slices."""
    dev_set = list(questions[:dev_size])
    test_set = list(questions[dev_size : dev_size + test_size]) if test_size else list(questions[dev_size:])
    return dev_set, test_set


def format_options(options: Sequence[str]) -> str:
    if not options:
        return "A. Not provided\nB. Not provided\nC. Not provided\nD. Not provided"
    return "\n".join(f"{chr(65 + index)}. {option}" for index, option in enumerate(options))


def get_correct_answer_letter(item: Dict) -> str:
    answer_index = item.get("answer_index", -1)
    if answer_index >= 0:
        return chr(65 + answer_index)
    return str(item.get("answer", "")).upper()


def extract_answer(response: str) -> Optional[str]:
    """Extract the final answer option from a model response."""
    if not response:
        return None

    strong_patterns = [
        r"(?i)answer\s*[:：]\s*([A-E])",
        r"(?i)correct\s*option\s*is\s*([A-E])",
        r"(?i)final\s*answer\s*[:：]\s*([A-E])",
    ]
    for pattern in strong_patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).upper()

    for pattern in (r"\*\*([A-E])\*\*", r"\(([A-E])\)", r"\[([A-E])\]"):
        matches = re.findall(pattern, response)
        if matches:
            return matches[-1].upper()

    fallback_matches = re.findall(r"\b([A-E])\b", response)
    return fallback_matches[-1].upper() if fallback_matches else None


class RateLimiter:
    """Token bucket rate limiter for async API calls."""

    def __init__(self, requests_per_second: float, burst: int = 10):
        self.requests_per_second = requests_per_second
        self.burst = burst
        self.tokens = burst
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.requests_per_second)
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

                wait_time = max((1 - self.tokens) / self.requests_per_second, 0.01)

            await asyncio.sleep(wait_time)


def create_async_client(config: EvaluationLLMConfig) -> AsyncOpenAI:
    """Create the shared async OpenAI-compatible client."""
    return AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=30.0,
        max_retries=2,
    )
