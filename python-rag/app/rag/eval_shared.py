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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    base_url: str = os.getenv("RAG_LLM_BASE_URL", DEFAULT_BASE_URL)
    api_key: str = os.getenv("RAG_LLM_API_KEY", DEFAULT_API_KEY)
    enable_thinking: Optional[bool] = field(
        default_factory=lambda: parse_optional_bool_env(
            "RAG_LLM_ENABLE_THINKING", default=False
        )
    )


def parse_optional_bool_env(
    name: str, default: Optional[bool] = None
) -> Optional[bool]:
    """Parse an optional boolean environment variable."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw_value}")


def build_extra_body(
    *,
    enable_thinking: Optional[bool] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build provider-specific request fields that must be sent in ``extra_body``."""
    payload = dict(extra_body or {})
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    return payload or None


@dataclass
class ConcurrencyConfig:
    rpm_limit: int = int(os.getenv("RAG_EVAL_RPM_LIMIT", "60"))
    max_concurrent: int = int(os.getenv("RAG_EVAL_MAX_CONCURRENT", "2"))

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
    test_set = (
        list(questions[dev_size : dev_size + test_size])
        if test_size
        else list(questions[dev_size:])
    )
    return dev_set, test_set


def format_options(options: Sequence[str]) -> str:
    if not options:
        return "A. Not provided\nB. Not provided\nC. Not provided\nD. Not provided"
    return "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
    )


def build_medical_eval_prompt(
    question: str, options: Sequence[str], context: Optional[str] = None
) -> str:
    """Build the shared evaluation prompt for both no-RAG and naive-RAG flows."""
    prompt_parts = ["You are a medical expert assistant."]
    if context:
        prompt_parts.append(
            "Answer the following question based on the provided context. "
            "If the context does not contain enough information to answer the question, "
            "state that you cannot answer based on the given information."
        )
        prompt_parts.extend(
            [
                "",
                "Context:",
                context,
            ]
        )
    else:
        prompt_parts.append(
            "Answer the following question based on your medical knowledge."
        )

    prompt_parts.extend(
        [
            "",
            f"Question: {question}",
            "",
            "Options:",
            format_options(options),
            "",
            "Provide only the final answer in the following format:",
            "Answer: [A/B/C/D/E]",
            "",
            "Your response:",
        ]
    )
    return "\n".join(prompt_parts)


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
                self.tokens = min(
                    self.burst, self.tokens + elapsed * self.requests_per_second
                )
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


def get_qwen_completion_kwargs(config: EvaluationLLMConfig) -> Dict[str, Any]:
    """Return the shared Qwen3-4B completion parameters."""
    kwargs = {
        "model": config.model,
        "temperature": config.temperature,
    }
    extra_body = build_extra_body(enable_thinking=config.enable_thinking)
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def get_qwen_langchain_kwargs(config: EvaluationLLMConfig) -> Dict[str, Any]:
    """Return the shared Qwen3-4B parameters for ChatOpenAI."""
    kwargs = {
        "model": config.model,
        "temperature": config.temperature,
    }
    extra_body = build_extra_body(enable_thinking=config.enable_thinking)
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


@dataclass
class EvalContext:
    """Shared evaluation context containing client and rate limiting primitives."""

    client: AsyncOpenAI
    semaphore: asyncio.Semaphore
    rate_limiter: RateLimiter
    llm_config: EvaluationLLMConfig


def create_eval_context(
    config: EvaluationLLMConfig, concurrency: ConcurrencyConfig
) -> EvalContext:
    """Create shared evaluation context with client and rate limiting."""
    return EvalContext(
        client=create_async_client(config),
        semaphore=asyncio.Semaphore(concurrency.max_concurrent),
        rate_limiter=RateLimiter(
            requests_per_second=concurrency.requests_per_second,
            burst=concurrency.max_concurrent,
        ),
        llm_config=config,
    )


async def call_llm(
    ctx: EvalContext,
    prompt: str,
) -> str:
    """Call LLM with rate limiting and return response content."""
    async with ctx.semaphore:
        await ctx.rate_limiter.acquire()
        completion = await ctx.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **get_qwen_completion_kwargs(ctx.llm_config),
        )
    return (
        completion.choices[0].message.content
        or completion.choices[0].message.reasoning_content
        or ""
    )


def build_eval_result(
    item: Dict[str, Any],
    response_content: str,
    rag_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build evaluation result dict from item and LLM response."""
    predicted_answer = extract_answer(response_content)
    correct_answer = get_correct_answer_letter(item)
    result = {
        "question": item["question"],
        "options": item.get("options", []),
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": predicted_answer == correct_answer,
        "response": response_content,
    }
    if rag_metadata:
        result.update(rag_metadata)
    return result


async def evaluate_single_item(
    ctx: EvalContext,
    item: Dict[str, Any],
    vectorstore: Optional[Any] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Evaluate a single question, optionally with RAG retrieval."""
    rag_metadata: Dict[str, Any] = {}
    if vectorstore:
        search_results = await asyncio.to_thread(
            vectorstore.similarity_search_with_score,
            item["question"],
            top_k,
        )
        docs = [doc for doc, _ in search_results]
        scores = [float(score) for _, score in search_results]
        contexts = [doc.page_content for doc in docs]
        context_str = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        prompt = build_medical_eval_prompt(
            question=item["question"],
            options=item.get("options", []),
            context=context_str,
        )
        rag_metadata = {
            "retrieved_docs": len(docs),
            "scores": scores,
            "contexts": contexts,
        }
    else:
        prompt = build_medical_eval_prompt(
            question=item["question"],
            options=item.get("options", []),
        )

    response_content = await call_llm(ctx, prompt)
    return build_eval_result(item, response_content, rag_metadata)


def update_progress(
    progress_mgr: Any,
    artifact_paths: Optional[Dict[str, Path]],
    live_config: Optional[Dict[str, Any]],
    extra_sections: Optional[Dict[str, Any]],
    dataset_name: str,
    total_questions: int,
    processed_questions: int,
    correct_count: int,
    elapsed: float,
    results: List[Dict[str, Any]],
    run_name: str,
    evaluation_type: str,
    config_payload: Dict[str, Any],
    script_name: str,
    top_k: Optional[int] = None,
) -> None:
    """Update progress checkpoint and live results if progress_mgr is provided."""
    if not progress_mgr:
        return

    progress_mgr.save_checkpoint(
        dataset_name=dataset_name,
        total_questions=total_questions,
        processed_questions=processed_questions,
        current_top_k=top_k or 0,
        results=results,
        correct_count=correct_count,
        total_count=processed_questions,
        elapsed_time=elapsed,
        config=config_payload,
        script_name=script_name,
    )
    stage_result = progress_mgr.build_stage_result(
        dataset_name=dataset_name,
        total_questions=total_questions,
        processed_questions=processed_questions,
        correct_count=correct_count,
        elapsed_time=elapsed,
        detailed_results=results,
        top_k=top_k,
    )
    progress_mgr.print_progress(
        run_name=run_name,
        dataset_name=dataset_name,
        processed_questions=processed_questions,
        total_questions=total_questions,
        correct_count=correct_count,
        elapsed_time=elapsed,
    )
    if artifact_paths and live_config:
        live_sections = dict(extra_sections or {})
        live_sections["current_stage"] = stage_result
        progress_mgr.write_live_results(
            artifact_paths=artifact_paths,
            run_name=run_name,
            evaluation_type=evaluation_type,
            config=live_config,
            stage_result=stage_result,
            extra_sections=live_sections,
        )
