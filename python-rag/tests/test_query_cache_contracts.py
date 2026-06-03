"""Query text and rewrite cache contracts for formal local embeddings."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from conftest import read_jsonl


def test_medcpt_query_specs_cover_naive_and_advanced_query_sources() -> None:
    from app.rag.experiments import run_medcpt_query_embedding_autodl as module

    specs_by_pipeline = {spec.pipeline: spec for spec in module.QUERY_EMBEDDING_SPECS}

    assert module.MEDCPT_QUERY_MODEL
    assert module.QUERY_INPUT_FORMAT == "retrieval_query_text_only"
    assert set(specs_by_pipeline) == {"naive_rag", "advanced_rag"}
    assert specs_by_pipeline["naive_rag"].query_text_source == (
        "medqa_usmle_question_field"
    )
    assert specs_by_pipeline["advanced_rag"].query_text_source == (
        "query_rewrite_pipeline"
    )


def test_medcpt_naive_query_text_rows_use_question_field_only() -> None:
    from app.rag.experiments import run_medcpt_query_embedding_autodl as module

    rows = module.build_naive_query_text_rows(
        [
            {
                "id": "dev-1",
                "question": "Which finding is most likely?",
                "options": ["Alpha", "Beta"],
            }
        ]
    )

    assert rows == [
        {
            "question_id": "dev-1",
            "question": "Which finding is most likely?",
            "query_text": "Which finding is most likely?",
            "query_text_source": "medqa_usmle_question_field",
            "contains_options": False,
            "contains_answer_prompt": False,
        }
    ]


def test_medcpt_advanced_query_embedding_requires_rewrite_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app.rag.experiments import run_medcpt_query_embedding_autodl as module

    spec = next(
        spec
        for spec in module.QUERY_EMBEDDING_SPECS
        if spec.pipeline == "advanced_rag"
    )
    monkeypatch.setattr(module, "_query_texts_path", lambda _: tmp_path / "missing.jsonl")

    with pytest.raises(FileNotFoundError) as exc_info:
        module.resolve_query_text_rows(spec, [{"id": "dev-1", "question": "Question?"}])

    assert "run_query_rewrite_cache_autodl.py" in str(exc_info.value)


def test_query_rewrite_cache_selects_only_advanced_specs() -> None:
    from app.rag.experiments import run_query_rewrite_cache_autodl as module

    specs = module._selected_rewrite_specs()

    assert specs
    assert {spec.pipeline for spec in specs} == {"advanced_rag"}


def test_advanced_query_text_rows_use_rewritten_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.rag.evaluation.eval_shared import EvaluationLLMConfig
    from app.rag.experiments import run_query_rewrite_cache_autodl as module

    class FakeRewritePipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def arewrite(self, query, **kwargs):
            return f"{query} rewritten for retrieval", [query]

    monkeypatch.setattr(module, "QueryRewritePipeline", FakeRewritePipeline)

    rows = asyncio.run(
        module.build_advanced_query_text_rows(
            [
                {
                    "id": "dev-1",
                    "question": "Which diagnosis is most likely?",
                    "options": ["Alpha", "Beta"],
                }
            ],
            EvaluationLLMConfig(),
        )
    )

    assert rows == [
        {
            "question_id": "dev-1",
            "question": "Which diagnosis is most likely?",
            "original_query": "Which diagnosis is most likely?",
            "query_text": "Which diagnosis is most likely? rewritten for retrieval",
            "query_text_source": "query_rewrite_pipeline",
            "contains_options": False,
            "contains_answer_prompt": False,
        }
    ]


def test_query_rewrite_cache_checkpoints_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app.rag.evaluation.eval_shared import EvaluationLLMConfig
    from app.rag.experiments import run_query_rewrite_cache_autodl as module
    from app.rag.experiments.run_medcpt_query_embedding_autodl import (
        QUERY_EMBEDDING_SPECS,
    )

    spec = next(
        spec for spec in QUERY_EMBEDDING_SPECS if spec.pipeline == "advanced_rag"
    )
    output_path = tmp_path / "query_texts.jsonl"
    questions = [
        {"id": "dev-1", "question": "Question one?"},
        {"id": "dev-2", "question": "Question two?"},
    ]

    class FailingRewritePipeline:
        async def arewrite(self, query, **kwargs):
            if query == "Question two?":
                raise RuntimeError("audit blocked")
            return f"{query} rewritten", [query]

    monkeypatch.setattr(module, "_query_texts_path", lambda _: output_path)
    monkeypatch.setattr(
        module,
        "create_query_rewriter",
        lambda llm_config: FailingRewritePipeline(),
    )

    monkeypatch.setattr(module, "RUN_MODE", "rewrite_all")
    asyncio.run(module.write_rewrite_cache(spec, questions, EvaluationLLMConfig()))

    checkpoint_path = output_path.with_name(module.QUERY_TEXTS_CHECKPOINT_FILENAME)
    errors_path = output_path.with_name(module.QUERY_REWRITE_ERRORS_FILENAME)
    assert [row["question_id"] for row in read_jsonl(checkpoint_path)] == ["dev-1"]
    assert read_jsonl(errors_path)[-1]["question_id"] == "dev-2"
    assert not output_path.exists()

    class SuccessfulRewritePipeline:
        async def arewrite(self, query, **kwargs):
            return f"{query} rewritten", [query]

    monkeypatch.setattr(
        module,
        "create_query_rewriter",
        lambda llm_config: SuccessfulRewritePipeline(),
    )
    monkeypatch.setattr(module, "RUN_MODE", "retry_errors")

    asyncio.run(module.write_rewrite_cache(spec, questions, EvaluationLLMConfig()))

    assert [row["question_id"] for row in read_jsonl(output_path)] == [
        "dev-1",
        "dev-2",
    ]
    assert not checkpoint_path.exists()


def test_query_rewrite_cache_fans_out_shared_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app.rag.evaluation.eval_shared import EvaluationLLMConfig
    from app.rag.experiments import run_query_rewrite_cache_autodl as module
    from app.rag.experiments.formal_query_embedding_specs import QueryEmbeddingSpec

    specs = [
        QueryEmbeddingSpec("advanced_a", "advanced_rag", "query_rewrite_pipeline"),
        QueryEmbeddingSpec("advanced_b", "advanced_rag", "query_rewrite_pipeline"),
    ]
    questions = [{"id": "dev-1", "question": "Question one?"}]
    output_paths = {
        "advanced_a": tmp_path / "advanced_a" / "query_texts.jsonl",
        "advanced_b": tmp_path / "advanced_b" / "query_texts.jsonl",
    }
    rewrite_calls = {"count": 0}

    class FakeRewritePipeline:
        async def arewrite(self, query, **kwargs):
            rewrite_calls["count"] += 1
            return f"{query} rewritten", [query]

    monkeypatch.setattr(module, "_query_texts_path", lambda spec: output_paths[spec.cache_id])
    monkeypatch.setattr(
        module,
        "create_query_rewriter",
        lambda llm_config: FakeRewritePipeline(),
    )
    monkeypatch.setattr(module, "RUN_MODE", "rewrite_all")

    asyncio.run(module.write_rewrite_caches(specs, questions, EvaluationLLMConfig()))

    assert rewrite_calls["count"] == 1
    assert read_jsonl(output_paths["advanced_a"]) == read_jsonl(
        output_paths["advanced_b"]
    )
