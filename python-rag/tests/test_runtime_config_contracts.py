"""Runtime configuration behavior contracts."""

from __future__ import annotations

import asyncio
import json


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
