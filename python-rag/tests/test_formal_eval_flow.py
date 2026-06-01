"""Tests for explicit formal evaluation flows."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.resolve()))


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

    query_rows = [
        json.loads(line)
        for line in (tmp_path / "retrieval" / "formal_naive" / "query_texts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    prompt_rows = [
        json.loads(line)
        for line in (tmp_path / "runs" / "formal_naive" / "final_prompts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert fake_store.queries == ["Which diagnosis is most likely?"]
    assert query_rows[0]["query_text"] == "Which diagnosis is most likely?"
    assert query_rows[0]["contains_options"] is False
    assert query_rows[0]["contains_answer_prompt"] is False
    assert "Options:" in prompt_rows[0]["prompt"]
    assert "context for Which diagnosis is most likely?" in prompt_rows[0]["prompt"]
    assert result["test_results"]["accuracy"] == 1.0


def test_enhanced_formal_writes_rewrite_and_component_caches(
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
            assert query.endswith("rewritten")
            return (
                [FakeNodeWithScore("dense context", 0.7)],
                [FakeNodeWithScore("sparse context", 0.6)],
                [FakeNodeWithScore("fusion context", 0.9)],
            )

    class FakeReranker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def rerank(self, query, documents):
            return documents

    async def fake_call_llm(ctx, prompt):
        return "Answer: A"

    monkeypatch.setattr(module, "load_vector_store", lambda _: FakeVectorStore())
    monkeypatch.setattr(module, "create_llm", lambda _: object())
    monkeypatch.setattr(module, "create_eval_context", lambda *args: SimpleNamespace(rate_limiter=None, semaphore=None))
    monkeypatch.setattr(module, "QueryRewritePipeline", FakeQueryRewritePipeline)
    monkeypatch.setattr(module, "HybridRetriever", FakeHybrid)
    monkeypatch.setattr(module, "RerankerPipeline", FakeReranker)
    monkeypatch.setattr(module, "call_llm", fake_call_llm)

    result = asyncio.run(
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

    query_rows = [
        json.loads(line)
        for line in (tmp_path / "retrieval" / "formal_advanced" / "query_texts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    fusion_rows = [
        json.loads(line)
        for line in (tmp_path / "retrieval" / "formal_advanced" / "fusion_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert query_rows[0]["query_text"] == "Which diagnosis is most likely? rewritten"
    assert query_rows[0]["contains_options"] is False
    assert fusion_rows[0]["candidate_source"] == "query_fusion"
    assert fusion_rows[0]["candidates"][0]["text"] == "fusion context"
    assert result["test_results"]["accuracy"] == 1.0


def test_medcpt_formal_retriever_consumes_autodl_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.rag.evaluation import formal_local_embedding_adapter as module
    from app.rag.evaluation.formal_local_embedding_adapter import LocalEmbeddingFormalRetriever

    index_root = tmp_path / "indexes" / "statpearls__ncbi_medcpt__FlatIP"
    query_root = tmp_path / "retrieval" / "stage1_naive_medcpt"
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
        query_cache_id="stage1_naive_medcpt",
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
    query_root = tmp_path / "retrieval" / "stage1_naive_medcpt"
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
        query_cache_id="stage1_naive_medcpt",
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
