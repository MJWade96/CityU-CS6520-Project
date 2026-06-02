"""
Shared helpers for baseline and naive RAG evaluation scripts.

Centralizing these utilities keeps prompt formatting, answer extraction,
dataset splitting, and rate limiting consistent across evaluation modes.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import openai
import httpx
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from openai import AsyncOpenAI

from ..data.benchmarks.medqa_usmle import load_medqa_usmle_jsonl
from ..data.data_paths import MEDQA_FILE
from ..data.json_utils import load_json_safe


T = TypeVar("T")
R = TypeVar("R")

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_PROVIDER = "Qwen3-8B"
DEFAULT_API_KEY = "sk-jwbxcbszdqdinhqofxikohzyjisdvwnkljbrzkfqufuxcbyy"


@dataclass
class EvaluationLLMConfig:
    provider: str = field(
        default_factory=lambda: os.getenv("RAG_LLM_PROVIDER", DEFAULT_PROVIDER)
    )
    model: str = field(
        default_factory=lambda: os.getenv("RAG_LLM_MODEL", DEFAULT_MODEL)
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("RAG_LLM_TEMPERATURE", "0.1"))
    )
    base_url: str = field(
        default_factory=lambda: os.getenv("RAG_LLM_BASE_URL", DEFAULT_BASE_URL)
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("RAG_LLM_API_KEY", DEFAULT_API_KEY)
    )
    enable_thinking: Optional[bool] = field(
        default_factory=lambda: parse_optional_bool_env(
            "RAG_LLM_ENABLE_THINKING", default=True
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
    rpm_limit: int = field(
        default_factory=lambda: int(os.getenv("RAG_EVAL_RPM_LIMIT", "1000"))
    )
    tpm_limit: int = field(
        default_factory=lambda: int(os.getenv("RAG_EVAL_TPM_LIMIT", "50000"))
    )
    max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("RAG_EVAL_MAX_CONCURRENT", "4"))
    )

    @property
    def requests_per_second(self) -> float:
        return self.rpm_limit / 60 * 0.9


def load_questions(question_file: Optional[str] = None) -> List[Dict]:
    """Load MedQA questions from JSON or the normalized MedQA-USMLE jsonl splits."""
    question_path = Path(question_file or MEDQA_FILE)
    if not question_path.exists():
        raise FileNotFoundError(f"Question file not found: {question_path}")

    if question_path.suffix == ".jsonl":
        return load_medqa_usmle_jsonl(question_path, split=question_path.stem)

    return load_json_safe(question_path)


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


def question_id(item: Mapping[str, Any], index: int, split: str = "dev") -> str:
    """Resolve stable question ids for formal artifacts and generated MedQA rows."""
    return str(item.get("id") or f"{split}-{index}")


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


def format_retrieved_contexts(contexts: Sequence[str]) -> str:
    """Format retrieved snippets once so formal Naive and Advanced prompts match."""
    return "\n\n".join(
        f"[{index}] {context}"
        for index, context in enumerate(contexts, start=1)
        if str(context).strip()
    )


def serialize_node_candidates(nodes: Sequence[Any]) -> List[Dict[str, Any]]:
    """Serialize LlamaIndex node scores once for formal retrieval artifacts."""
    rows: List[Dict[str, Any]] = []
    for rank, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        rows.append(
            {
                "rank": rank,
                "score": float(node_with_score.score or 0.0),
                "text": node.get_content(),
                "metadata": dict(node.metadata),
            }
        )
    return rows


def serialize_document_candidates(
    documents: Sequence[Tuple[Any, float]],
) -> List[Dict[str, Any]]:
    """Serialize ``RetrievedDocument`` score pairs for formal artifacts."""
    return [
        {
            "rank": rank,
            "score": float(score),
            "text": document.page_content,
            "metadata": dict(document.metadata),
        }
        for rank, (document, score) in enumerate(documents, start=1)
    ]


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


async def iter_pipeline_in_order(
    items: Sequence[T],
    *,
    max_concurrent: int,
    worker: Callable[[int, T], Awaitable[R]],
    start_index: int = 0,
    heartbeat_interval: Optional[float] = None,
    on_heartbeat: Optional[Callable[[], None]] = None,
) -> AsyncIterator[Tuple[int, T, R]]:
    """Run work as a refill pipeline while yielding results in input order.

    This keeps checkpoint updates prefix-based while avoiding batch-level idle slots.
    """
    if not items:
        return

    limit = max(1, max_concurrent)
    next_offset = 0
    commit_offset = 0
    pending: Dict[asyncio.Task[R], int] = {}
    completed: Dict[int, R] = {}

    def schedule_available() -> None:
        nonlocal next_offset
        while next_offset < len(items) and len(pending) < limit:
            offset = next_offset
            pending[
                asyncio.create_task(worker(start_index + offset, items[offset]))
            ] = offset
            next_offset += 1

    async def cancel_pending() -> None:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending.keys(), return_exceptions=True)

    schedule_available()
    while commit_offset < len(items):
        while commit_offset in completed:
            result = completed.pop(commit_offset)
            yield start_index + commit_offset, items[commit_offset], result
            commit_offset += 1

        if commit_offset >= len(items):
            break

        done, _ = await asyncio.wait(
            pending.keys(),
            timeout=heartbeat_interval,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            if on_heartbeat is not None:
                on_heartbeat()
            continue

        for task in done:
            offset = pending.pop(task)
            try:
                completed[offset] = task.result()
            except Exception:
                await cancel_pending()
                raise

        schedule_available()


class TokenRateLimiter:
    """Token bucket limiter for estimated prompt plus completion budget."""

    def __init__(self, tokens_per_second: float, burst: int):
        self.tokens_per_second = tokens_per_second
        self.burst = max(1, burst)
        self.tokens = self.burst
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, token_count: int) -> None:
        requested = max(1, min(token_count, self.burst))
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(
                    self.burst, self.tokens + elapsed * self.tokens_per_second
                )
                self.last_update = now

                if self.tokens >= requested:
                    self.tokens -= requested
                    return

                wait_time = max(
                    (requested - self.tokens) / self.tokens_per_second, 0.01
                )

            await asyncio.sleep(wait_time)


def create_async_client(config: EvaluationLLMConfig) -> AsyncOpenAI:
    """Create the shared async OpenAI-compatible client."""
    timeout = float(os.getenv("RAG_LLM_TIMEOUT", "120.0"))
    max_retries = int(os.getenv("RAG_LLM_MAX_RETRIES", "5"))
    return AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_qwen_completion_kwargs(config: EvaluationLLMConfig) -> Dict[str, Any]:
    """Return the shared Qwen completion parameters."""
    kwargs = {
        "model": config.model,
        "temperature": config.temperature,
    }
    extra_body = build_extra_body(enable_thinking=config.enable_thinking)
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def get_qwen_openai_like_kwargs(config: EvaluationLLMConfig) -> Dict[str, Any]:
    """Return the shared Qwen parameters for LlamaIndex OpenAILike."""
    kwargs: Dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "api_key": config.api_key,
        "api_base": config.base_url,
        "is_chat_model": True,
        "timeout": float(os.getenv("RAG_LLM_TIMEOUT", "120.0")),
        "max_retries": int(os.getenv("RAG_LLM_MAX_RETRIES", "5")),
    }
    extra_body = build_extra_body(enable_thinking=config.enable_thinking)
    if extra_body:
        # LlamaIndex flattens additional_kwargs into the OpenAI SDK call, so
        # provider-specific fields must stay nested under extra_body.
        kwargs["additional_kwargs"] = {"extra_body": extra_body}
    return kwargs


@dataclass
class EvalContext:
    """Shared evaluation context containing client and rate limiting primitives."""

    client: AsyncOpenAI
    semaphore: asyncio.Semaphore
    rate_limiter: RateLimiter
    token_rate_limiter: Optional[TokenRateLimiter]
    llm_config: EvaluationLLMConfig


def create_eval_context(
    config: EvaluationLLMConfig, concurrency: ConcurrencyConfig
) -> EvalContext:
    """Create shared evaluation context with client and rate limiting."""
    token_rate_limiter = None
    if concurrency.tpm_limit > 0:
        token_rate_limiter = TokenRateLimiter(
            tokens_per_second=concurrency.tpm_limit / 60 * 0.9,
            burst=concurrency.tpm_limit,
        )
    return EvalContext(
        client=create_async_client(config),
        semaphore=asyncio.Semaphore(concurrency.max_concurrent),
        rate_limiter=RateLimiter(
            requests_per_second=concurrency.requests_per_second,
            burst=concurrency.max_concurrent,
        ),
        token_rate_limiter=token_rate_limiter,
        llm_config=config,
    )


def estimate_llm_request_tokens(prompt: str) -> int:
    """Conservatively estimate one chat request for TPM throttling."""
    completion_reserve = int(os.getenv("RAG_LLM_COMPLETION_TOKEN_RESERVE", "512"))
    return max(1, int(len(prompt) / 4)) + max(0, completion_reserve)


async def call_llm(
    ctx: EvalContext,
    prompt: str,
) -> str:
    """Call LLM with rate limiting and return response content."""
    import asyncio

    max_retries = int(os.getenv("RAG_LLM_MAX_RETRIES", "5"))
    base_delay = 1.0
    last_exception = None

    for attempt in range(max_retries):
        try:
            async with ctx.semaphore:
                await ctx.rate_limiter.acquire()
                if ctx.token_rate_limiter:
                    await ctx.token_rate_limiter.acquire(
                        estimate_llm_request_tokens(prompt)
                    )
                completion = await ctx.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    **get_qwen_completion_kwargs(ctx.llm_config),
                )
            return (
                completion.choices[0].message.content
                or completion.choices[0].message.reasoning_content
                or ""
            )
        except (
            # 捕获 OpenAI SDK 抛出的主要异常
            openai.RateLimitError,  # 429 限流
            openai.InternalServerError,  # 500 服务器错误
            openai.APIConnectionError,  # 网络连接中断
            openai.APIStatusError,  # 其他非 200 状态码
            # 兼容底层 httpx 请求错误
            httpx.RequestError,
        ) as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)  # Exponential backoff
                # 建议加一行日志，方便观察重试状态
                # print(f"API Warning: {type(e).__name__} encountered. Retrying in {delay}s...")
                await asyncio.sleep(delay)
            continue

    raise last_exception


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
