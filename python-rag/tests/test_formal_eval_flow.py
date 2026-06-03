"""Tests for explicit formal evaluation flows."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import read_jsonl


class FakeNode:
    def __init__(self, text: str, metadata: dict | None = None):
        self._text = text
        self.metadata = metadata or {}

    def get_content(self) -> str:
        return self._text


class FakeNodeWithScore:
    def __init__(self, text: str, score: float = 0.9):
        self.node = FakeNode(text, {"source": "fake"})
        self.score = score


class FakeVectorStore:
    def __init__(self):
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int):
        self.queries.append(query)
        return [FakeNodeWithScore(f"context for {query}", 0.8)]

    def as_query_engine(self, **kwargs):
        raise AssertionError("formal mode must not use query engine")


def write_questions(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "dev-1",
                    "question": "Which diagnosis is most likely?",
                    "options": ["Alpha", "Beta"],
                    "answer_index": 0,
                }
            ]
        ),
        encoding="utf-8",
    )


def write_two_questions(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "dev-1",
                    "question": "Which diagnosis is most likely?",
                    "options": ["Alpha", "Beta"],
                    "answer_index": 0,
                },
                {
                    "id": "dev-2",
                    "question": "Which treatment is best?",
                    "options": ["Alpha", "Beta"],
                    "answer_index": 0,
                },
            ]
        ),
        encoding="utf-8",
    )


def patch_formal_dirs(monkeypatch, tmp_path: Path) -> None:
    from app.rag.evaluation import formal_artifacts

    monkeypatch.setattr(formal_artifacts, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(formal_artifacts, "RETRIEVAL_CACHE_DIR", tmp_path / "retrieval")
    monkeypatch.setattr(formal_artifacts, "RERANK_CACHE_DIR", tmp_path / "rerank")


def test_naive_formal_uses_question_text_for_retrieval(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.rag.evaluation import naive_rag_eval as module
    from app.rag.evaluation.config import NaiveRAGEvalConfig

    patch_formal_dirs(monkeypatch, tmp_path)
    question_file = tmp_path / "questions.json"
    write_questions(question_file)
    fake_store = FakeVectorStore()
    monkeypatch.setattr(module, "load_vector_store", lambda _: fake_store)
    monkeypatch.setattr(module, "create_eval_context", lambda *args: object())

    async def fake_call_llm(ctx, prompt):
        return "Answer: A"

    monkeypatch.setattr(module, "call_llm", fake_call_llm)
    result = asyncio.run(
        module.run_complete_evaluation(
            NaiveRAGEvalConfig(
                dev_size=0,
                test_size=1,
                manual_top_k=1,
                question_file=question_file,
                vector_store_path=tmp_path / "index",
                formal_run_id="formal_naive",
                formal_metadata={
                    "run_id": "formal_naive",
                    "pipeline": "naive_rag",
                    "embedding_backend": "siliconflow_api",
                    "query_cache_id": "formal_naive",
                },
            )
        )
    )

    query_rows = read_jsonl(
        tmp_path / "retrieval" / "formal_naive" / "query_texts.jsonl"
    )
    prompt_rows = read_jsonl(
        tmp_path / "runs" / "formal_naive" / "final_prompts.jsonl"
    )

    assert fake_store.queries == ["Which diagnosis is most likely?"]
    assert query_rows[0]["query_text"] == "Which diagnosis is most likely?"
    assert query_rows[0]["contains_options"] is False
    assert query_rows[0]["contains_answer_prompt"] is False
    assert "Options:" in prompt_rows[0]["prompt"]
    assert "context for Which diagnosis is most likely?" in prompt_rows[0]["prompt"]
    assert result["test_results"]["accuracy"] == 1.0


def test_naive_formal_records_generator_errors_without_stopping_pipeline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.rag.evaluation import naive_rag_eval as module
    from app.rag.evaluation.config import NaiveRAGEvalConfig
    from app.rag.evaluation.eval_shared import ConcurrencyConfig

    patch_formal_dirs(monkeypatch, tmp_path)
    question_file = tmp_path / "questions.json"
    write_two_questions(question_file)
    monkeypatch.setattr(module, "load_vector_store", lambda _: FakeVectorStore())
    monkeypatch.setattr(module, "create_eval_context", lambda *args: object())

    async def fake_call_llm(ctx, prompt):
        if "Which diagnosis is most likely?" in prompt:
            raise RuntimeError("rate limited")
        return "Answer: A"

    monkeypatch.setattr(module, "call_llm", fake_call_llm)

    result = asyncio.run(
        module.run_complete_evaluation(
            NaiveRAGEvalConfig(
                dev_size=0,
                test_size=2,
                manual_top_k=1,
                question_file=question_file,
                vector_store_path=tmp_path / "index",
                concurrency=ConcurrencyConfig(max_concurrent=2),
                formal_run_id="formal_naive",
                formal_metadata={
                    "run_id": "formal_naive",
                    "pipeline": "naive_rag",
                    "embedding_backend": "siliconflow_api",
                    "query_cache_id": "formal_naive",
                },
            )
        )
    )

    run_dir = tmp_path / "runs" / "formal_naive"
    error_rows = read_jsonl(run_dir / "generator_errors.jsonl")
    llm_rows = read_jsonl(run_dir / "llm_outputs.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert result["test_results"]["status"] == "generator_errors"
    assert result["test_results"]["processed_questions"] == 1
    assert result["test_results"]["failed_generator_questions"] == 1
    assert error_rows == [
        {
            "question_id": "dev-1",
            "error_type": "RuntimeError",
            "error_message": "rate limited",
        }
    ]
    assert [row["question_id"] for row in llm_rows] == ["dev-2"]
    assert manifest["status"] == "generator_errors"


def test_naive_formal_resume_reuses_partial_question_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.rag.evaluation import naive_rag_eval as module
    from app.rag.evaluation.config import NaiveRAGEvalConfig

    patch_formal_dirs(monkeypatch, tmp_path)
    question_file = tmp_path / "questions.json"
    write_questions(question_file)
    fake_store = FakeVectorStore()
    monkeypatch.setattr(module, "load_vector_store", lambda _: fake_store)
    monkeypatch.setattr(module, "create_eval_context", lambda *args: object())

    retrieval_dir = tmp_path / "retrieval" / "formal_naive"
    run_dir = tmp_path / "runs" / "formal_naive"
    retrieval_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    cached_candidates = [{"text": "cached context", "score": 0.7}]
    (retrieval_dir / "query_texts.jsonl").write_text(
        json.dumps(
            {
                "question_id": "dev-1",
                "question": "Which diagnosis is most likely?",
                "query_text": "Which diagnosis is most likely?",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (retrieval_dir / "retrieval_top10.jsonl").write_text(
        json.dumps({"question_id": "dev-1", "candidates": cached_candidates}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "selected_contexts.jsonl").write_text(
        json.dumps({"question_id": "dev-1", "selected_contexts": cached_candidates}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "final_prompts.jsonl").write_text(
        json.dumps({"question_id": "dev-1", "prompt": "cached prompt"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "llm_outputs.jsonl").write_text(
        json.dumps({"question_id": "dev-1", "response": "Answer: A"}) + "\n",
        encoding="utf-8",
    )

    async def fail_if_called(ctx, prompt):
        raise AssertionError("resume should reuse cached LLM output")

    monkeypatch.setattr(module, "call_llm", fail_if_called)

    result = asyncio.run(
        module.run_complete_evaluation(
            NaiveRAGEvalConfig(
                dev_size=0,
                test_size=1,
                manual_top_k=1,
                question_file=question_file,
                vector_store_path=tmp_path / "index",
                formal_run_id="formal_naive",
                formal_metadata={
                    "run_id": "formal_naive",
                    "pipeline": "naive_rag",
                    "embedding_backend": "siliconflow_api",
                    "query_cache_id": "formal_naive",
                },
            )
        )
    )

    assert fake_store.queries == []
    assert result["test_results"]["processed_questions"] == 1
    for path in (
        retrieval_dir / "query_texts.jsonl",
        retrieval_dir / "retrieval_top10.jsonl",
        run_dir / "selected_contexts.jsonl",
        run_dir / "final_prompts.jsonl",
        run_dir / "llm_outputs.jsonl",
        run_dir / "evaluation_outputs.jsonl",
    ):
        assert len(read_jsonl(path)) == 1


def test_enhanced_formal_writes_rewrite_and_component_caches(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.rag.evaluation import enhanced_rag_eval as module
    from app.rag.evaluation.enhanced_rag_eval import EnhancedEvaluationConfig
    from app.rag.evaluation.formal_local_rerank_cache import (
        LOCAL_RERANKER_BACKEND,
        rerank_cache_id,
    )

    patch_formal_dirs(monkeypatch, tmp_path)
    question_file = tmp_path / "questions.json"
    write_questions(question_file)

    class FakeQueryRewritePipeline:
        dict_rewriter = SimpleNamespace(ABBREVIATIONS={}, CHINESE_TERMS={})
        llm_rewriter = object()

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def arewrite(self, query, **kwargs):
            return f"{query} rewritten", ["llm"]

    class FakeHybrid:
        @classmethod
        def from_vector_store(cls, *args, **kwargs):
            return cls()

        def retrieve_components(self, query: str):
            assert query.endswith("rewritten")
            return (
                [FakeNodeWithScore("dense context", 0.7)],
                [FakeNodeWithScore("sparse context", 0.6)],
                [FakeNodeWithScore("fusion context", 0.9)],
            )

    async def fake_call_llm(ctx, prompt):
        return "Answer: A"

    config = EnhancedEvaluationConfig(
        dev_size=0,
        test_size=1,
        top_k=1,
        retrieval_top_k=2,
        reranker_top_k=1,
        question_file=question_file,
        vector_store_path=tmp_path / "index",
        formal_run_id="formal_advanced",
        formal_metadata={
            "run_id": "formal_advanced",
            "pipeline": "advanced_rag",
            "embedding_backend": "siliconflow_api",
            "query_cache_id": "formal_advanced",
        },
    )
    rerank_dir = tmp_path / "rerank" / rerank_cache_id(
        retrieval_candidates_id="formal_advanced:fusion_candidates",
        reranker_model=config.reranker_model,
        reranker_input_count=2,
    )
    rerank_dir.mkdir(parents=True)
    (rerank_dir / "rerank_outputs.jsonl").write_text(
        json.dumps(
            {
                "question_id": "dev-1",
                "input_candidates_id": "dev-1:fusion_candidates",
                "reranker_backend": LOCAL_RERANKER_BACKEND,
                "reranker_model": config.reranker_model,
                "reranker_input_count": 2,
                "reranker_output_count": 1,
                "reranked_candidates": [
                    {
                        "rank": 1,
                        "score": 1.0,
                        "text": "fusion context",
                        "metadata": {"source": "fake"},
                    }
                ],
                "rerank_time_seconds": 0.01,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "load_vector_store", lambda _: FakeVectorStore())
    monkeypatch.setattr(module, "create_llm", lambda _: object())
    monkeypatch.setattr(module, "create_eval_context", lambda *args: SimpleNamespace(rate_limiter=None, semaphore=None))
    monkeypatch.setattr(module, "QueryRewritePipeline", FakeQueryRewritePipeline)
    monkeypatch.setattr(module, "HybridRetriever", FakeHybrid)
    monkeypatch.setattr(module, "RerankerPipeline", lambda **kwargs: (_ for _ in ()).throw(AssertionError("formal mode must not call API reranker")))
    monkeypatch.setattr(module, "call_llm", fake_call_llm)

    result = asyncio.run(module.run_enhanced_evaluation(config))

    query_rows = read_jsonl(
        tmp_path / "retrieval" / "formal_advanced" / "query_texts.jsonl"
    )
    fusion_rows = read_jsonl(
        tmp_path / "retrieval" / "formal_advanced" / "fusion_candidates.jsonl"
    )

    assert query_rows[0]["query_text"] == "Which diagnosis is most likely? rewritten"
    assert query_rows[0]["contains_options"] is False
    assert fusion_rows[0]["candidate_source"] == "query_fusion"
    assert fusion_rows[0]["candidates"][0]["text"] == "fusion context"
    assert result["test_results"]["accuracy"] == 1.0


def test_enhanced_formal_requires_local_rerank_cache_after_retrieval(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.rag.evaluation import enhanced_rag_eval as module
    from app.rag.evaluation.enhanced_rag_eval import EnhancedEvaluationConfig

    patch_formal_dirs(monkeypatch, tmp_path)
    question_file = tmp_path / "questions.json"
    write_questions(question_file)

    class FakeQueryRewritePipeline:
        dict_rewriter = SimpleNamespace(ABBREVIATIONS={}, CHINESE_TERMS={})
        llm_rewriter = object()

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def arewrite(self, query, **kwargs):
            return f"{query} rewritten", ["llm"]

    class FakeHybrid:
        @classmethod
        def from_vector_store(cls, *args, **kwargs):
            return cls()

        def retrieve_components(self, query: str):
            return (
                [FakeNodeWithScore("dense context", 0.7)],
                [FakeNodeWithScore("sparse context", 0.6)],
                [FakeNodeWithScore("fusion context", 0.9)],
            )

    monkeypatch.setattr(module, "load_vector_store", lambda _: FakeVectorStore())
    monkeypatch.setattr(module, "create_llm", lambda _: object())
    monkeypatch.setattr(module, "create_eval_context", lambda *args: SimpleNamespace(rate_limiter=None, semaphore=None))
    monkeypatch.setattr(module, "QueryRewritePipeline", FakeQueryRewritePipeline)
    monkeypatch.setattr(module, "HybridRetriever", FakeHybrid)
    monkeypatch.setattr(module, "RerankerPipeline", lambda **kwargs: (_ for _ in ()).throw(AssertionError("formal mode must not call API reranker")))

    with pytest.raises(FileNotFoundError, match="run_local_rerank_cache_autodl"):
        asyncio.run(
            module.run_enhanced_evaluation(
                EnhancedEvaluationConfig(
                    dev_size=0,
                    test_size=1,
                    top_k=1,
                    retrieval_top_k=2,
                    reranker_top_k=1,
                    question_file=question_file,
                    vector_store_path=tmp_path / "index",
                    formal_run_id="formal_advanced",
                    formal_metadata={
                        "run_id": "formal_advanced",
                        "pipeline": "advanced_rag",
                        "embedding_backend": "siliconflow_api",
                        "query_cache_id": "formal_advanced",
                    },
                )
            )
        )

    assert (
        tmp_path / "retrieval" / "formal_advanced" / "fusion_candidates.jsonl"
    ).exists()


def test_medcpt_formal_retriever_consumes_autodl_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.rag.evaluation import formal_local_embedding_adapter as module
    from app.rag.evaluation.formal_local_embedding_adapter import LocalEmbeddingFormalRetriever

    index_root = tmp_path / "indexes" / "statpearls__ncbi_medcpt__FlatIP"
    query_cache_id = "local-query-cache"
    query_root = tmp_path / "retrieval" / query_cache_id
    index_root.mkdir(parents=True)
    query_root.mkdir(parents=True)
    np.save(index_root / "chunk_embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    np.save(query_root / "query_embeddings.npy", np.asarray([[0.0, 1.0]], dtype="float32"))
    (index_root / "manifest.json").write_text(
        json.dumps({"selected_sources": ["statpearls"]}),
        encoding="utf-8",
    )
    (query_root / "query_texts.jsonl").write_text(
        json.dumps({"question_id": "dev-1", "query_text": "query"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "RETRIEVAL_CACHE_DIR", tmp_path / "retrieval")
    monkeypatch.setattr(
        module,
        "combine_registered_corpora",
        lambda selected_sources: {
            "records": [
                {"id": "doc-1", "contents": "first", "source": "statpearls"},
                {"id": "doc-2", "contents": "second", "source": "statpearls"},
            ]
        },
    )

    retriever = LocalEmbeddingFormalRetriever.load(
        corpus_version="statpearls",
        index_root=index_root,
        query_cache_id=query_cache_id,
    )

    results = retriever.retrieve(question_id="dev-1", query_text="query", k=1)

    assert results[0][0].page_content == "second"
    assert results[0][1] == pytest.approx(1.0)


def test_medcpt_formal_retriever_components_require_explicit_llm(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from llama_index.core.llms.mock import MockLLM

    from app.rag.evaluation import formal_local_embedding_adapter as module
    from app.rag.evaluation.formal_local_embedding_adapter import LocalEmbeddingFormalRetriever

    index_root = tmp_path / "indexes" / "statpearls__ncbi_medcpt__FlatIP"
    query_cache_id = "local-query-cache"
    query_root = tmp_path / "retrieval" / query_cache_id
    index_root.mkdir(parents=True)
    query_root.mkdir(parents=True)
    np.save(index_root / "chunk_embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    np.save(query_root / "query_embeddings.npy", np.asarray([[0.0, 1.0]], dtype="float32"))
    (index_root / "manifest.json").write_text(
        json.dumps({"selected_sources": ["statpearls"]}),
        encoding="utf-8",
    )
    (query_root / "query_texts.jsonl").write_text(
        json.dumps({"question_id": "dev-1", "query_text": "second"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "RETRIEVAL_CACHE_DIR", tmp_path / "retrieval")
    monkeypatch.setattr(
        module,
        "combine_registered_corpora",
        lambda selected_sources: {
            "records": [
                {"id": "doc-1", "contents": "first text", "source": "statpearls"},
                {"id": "doc-2", "contents": "second text", "source": "statpearls"},
            ]
        },
    )
    retriever = LocalEmbeddingFormalRetriever.load(
        corpus_version="statpearls",
        index_root=index_root,
        query_cache_id=query_cache_id,
    )

    with pytest.raises(ValueError, match="explicit LlamaIndex LLM"):
        retriever.retrieve_components(
            question_id="dev-1",
            query_text="second",
            k=1,
            weights=(0.5, 0.5),
            llm=None,
        )

    dense, sparse, fusion = retriever.retrieve_components(
        question_id="dev-1",
        query_text="second",
        k=1,
        weights=(0.5, 0.5),
        llm=MockLLM(),
    )

    assert dense
    assert sparse
    assert fusion
