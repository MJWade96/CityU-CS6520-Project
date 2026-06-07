"""Behavior contracts for LLM retry outcomes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest


class StubCompletions:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    async def create(self, **_kwargs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def bad_request(error_type: str) -> openai.BadRequestError:
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://example.test/chat"),
    )
    return openai.BadRequestError(
        error_type,
        response=response,
        body={"error": {"type": error_type}},
    )


def run_call(monkeypatch, outcomes, *, retries=5):
    from app.rag.evaluation import eval_shared

    async def no_wait(*_args):
        return None

    completions = StubCompletions(outcomes)
    ctx = SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        semaphore=asyncio.Semaphore(1),
        rate_limiter=SimpleNamespace(acquire=no_wait),
        token_rate_limiter=None,
        llm_config=eval_shared.EvaluationLLMConfig(api_key="test"),
        rate_limit_cooldown_lock=asyncio.Lock(),
        rate_limit_cooldown_until=0.0,
    )
    monkeypatch.setenv("RAG_LLM_MAX_RETRIES", str(retries))
    monkeypatch.setattr(eval_shared.asyncio, "sleep", no_wait)
    return eval_shared, completions, asyncio.run(eval_shared.call_llm(ctx, "prompt"))


def test_output_audit_retries_once_then_returns_marker(monkeypatch) -> None:
    marker = "TEXT_AUDIT_ANSWER_NOT_PASS"
    module, calls, result = run_call(
        monkeypatch,
        [bad_request(marker), bad_request(marker)],
    )

    assert (calls.calls, result) == (2, module.TEXT_AUDIT_ANSWER_NOT_PASS)


def test_output_audit_can_succeed_on_its_single_retry(monkeypatch) -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Answer: A"), finish_reason="stop")]
    )
    _, calls, result = run_call(
        monkeypatch,
        [bad_request("TEXT_AUDIT_ANSWER_NOT_PASS"), completion],
    )

    assert (calls.calls, result) == (2, "Answer: A")


def test_other_bad_requests_keep_configured_retry_count(monkeypatch) -> None:
    error = bad_request("OTHER_BAD_REQUEST")

    with pytest.raises(openai.BadRequestError):
        run_call(monkeypatch, [error, error, error], retries=3)
