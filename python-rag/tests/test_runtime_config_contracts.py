"""Runtime configuration behavior contracts."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import openai
import pytest


def test_openai_like_kwargs_keep_enable_thinking_inside_extra_body() -> None:
    from llama_index.llms.openai_like import OpenAILike

    from app.rag.evaluation.eval_shared import (
        EvaluationLLMConfig,
        get_qwen_openai_like_kwargs,
    )

    kwargs = get_qwen_openai_like_kwargs(EvaluationLLMConfig(enable_thinking=False))
    model_kwargs = OpenAILike(**kwargs)._get_model_kwargs(max_tokens=7)

    assert kwargs["additional_kwargs"] == {"extra_body": {"enable_thinking": False}}
    assert "enable_thinking" not in kwargs["additional_kwargs"]
    assert model_kwargs["extra_body"] == {"enable_thinking": False}
    assert "enable_thinking" not in model_kwargs


def test_llm_clients_ignore_environment_proxies_by_default() -> None:
    from app.rag.evaluation.eval_shared import (
        EvaluationLLMConfig,
        create_async_client,
        get_qwen_openai_like_kwargs,
    )

    config = EvaluationLLMConfig(api_key="test-key")
    client = create_async_client(config)
    kwargs = get_qwen_openai_like_kwargs(config)

    try:
        assert getattr(client._client, "_trust_env") is False
        assert getattr(kwargs["http_client"], "_trust_env") is False
        assert getattr(kwargs["async_http_client"], "_trust_env") is False
    finally:
        asyncio.run(client.close())
        kwargs["http_client"].close()
        asyncio.run(kwargs["async_http_client"].aclose())


def test_llm_config_reads_environment_at_instantiation(monkeypatch) -> None:
    from app.rag.evaluation.eval_shared import EvaluationLLMConfig

    monkeypatch.setenv("RAG_LLM_MODEL", "env-model")
    monkeypatch.setenv("RAG_LLM_API_KEY", "env-key")

    llm = EvaluationLLMConfig()

    assert llm.model == "env-model"
    assert llm.api_key == "env-key"


def test_resolve_embedding_runtime_prefers_recorded_model_over_env(
    monkeypatch, tmp_path
) -> None:
    metadata_path = tmp_path / "build_metadata.json"
    metadata_path.write_text(
        json.dumps({"embedding_model": "recorded-model"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "env-model")

    from app.rag.retriever.runtime_config import resolve_embedding_runtime

    runtime = resolve_embedding_runtime(
        str(tmp_path),
        default_model="default-model",
    )

    assert runtime["model_name"] == "recorded-model"


class _ImmediateRateLimiter:
    async def acquire(self) -> None:
        return None


class _SequenceCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def create(self, **kwargs):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _bad_request(error_type: str) -> openai.BadRequestError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(400, request=request)
    body = {"error": {"type": error_type}}
    return openai.BadRequestError(
        f"Error code: 400 - {body}",
        response=response,
        body=body,
    )


def _eval_context(outcomes):
    from app.rag.evaluation.eval_shared import EvalContext, EvaluationLLMConfig

    completions = _SequenceCompletions(outcomes)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    context = EvalContext(
        client=client,
        semaphore=asyncio.Semaphore(1),
        rate_limiter=_ImmediateRateLimiter(),
        token_rate_limiter=None,
        llm_config=EvaluationLLMConfig(api_key="test-key"),
    )
    return context, completions


def test_text_audit_answer_error_retries_once_without_delay(monkeypatch) -> None:
    from app.rag.evaluation import eval_shared

    audit_error = _bad_request("TEXT_AUDIT_ANSWER_NOT_PASS")
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Answer: A"),
                finish_reason="stop",
            )
        ]
    )
    context, completions = _eval_context([audit_error, completion])

    async def fail_if_slept(delay):
        raise AssertionError("output-audit retry must not use backoff")

    monkeypatch.setattr(eval_shared.asyncio, "sleep", fail_if_slept)

    result = asyncio.run(eval_shared.call_llm(context, "prompt"))

    assert result == "Answer: A"
    assert completions.calls == 2


def test_text_audit_answer_error_returns_marker_after_one_extra_attempt(
    monkeypatch,
) -> None:
    from app.rag.evaluation import eval_shared

    context, completions = _eval_context(
        [
            _bad_request("TEXT_AUDIT_ANSWER_NOT_PASS"),
            _bad_request("TEXT_AUDIT_ANSWER_NOT_PASS"),
        ]
    )

    async def fail_if_slept(delay):
        raise AssertionError("output-audit retry must not use backoff")

    monkeypatch.setattr(eval_shared.asyncio, "sleep", fail_if_slept)

    result = asyncio.run(eval_shared.call_llm(context, "prompt"))

    assert result == eval_shared.TEXT_AUDIT_ANSWER_NOT_PASS
    assert completions.calls == 2


def test_other_bad_request_keeps_existing_retry_policy(monkeypatch) -> None:
    from app.rag.evaluation import eval_shared

    monkeypatch.setenv("RAG_LLM_MAX_RETRIES", "3")
    context, completions = _eval_context(
        [
            _bad_request("OTHER_BAD_REQUEST"),
            _bad_request("OTHER_BAD_REQUEST"),
            _bad_request("OTHER_BAD_REQUEST"),
        ]
    )
    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(eval_shared.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(eval_shared.random, "uniform", lambda *_: 0.0)

    with pytest.raises(openai.BadRequestError, match="OTHER_BAD_REQUEST"):
        asyncio.run(eval_shared.call_llm(context, "prompt"))

    assert completions.calls == 3
    assert delays == [2.0, 4.0]
