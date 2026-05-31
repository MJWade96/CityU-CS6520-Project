"""Focused tests for phase 1 corpus and ablation plumbing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.resolve()))


def test_corpus_registry_normalizes_medrag_sources(tmp_path: Path) -> None:
    from app.rag.data.corpus_registry import combine_registered_corpora

    statpearls_file = tmp_path / "statpearls.json"
    textbooks_file = tmp_path / "textbooks.jsonl"
    statpearls_file.write_text(
        json.dumps(
            [
                {
                    "id": "sp-1",
                    "title": "StatPearls title",
                    "content": "StatPearls content",
                    "contents": "StatPearls title. StatPearls content",
                }
            ]
        ),
        encoding="utf-8",
    )
    textbooks_file.write_text(
        json.dumps(
            {
                "id": "tb-1",
                "title": "Textbook title",
                "content": "Textbook content",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = combine_registered_corpora(
        source_files={
            "statpearls": str(statpearls_file),
            "textbooks": str(textbooks_file),
        },
        selected_sources=["statpearls", "textbooks"],
    )

    records = result["records"]
    assert [record["source"] for record in records] == ["statpearls", "textbooks"]
    assert records[1]["contents"] == "Textbook title. Textbook content"
    assert result["stats"]["textbooks"]["count"] == 1


def test_corpus_registry_uses_existing_combined_corpus_for_statpearls() -> None:
    from app.rag.data.corpus_registry import CORPUS_REGISTRY
    from app.rag.data.data_paths import COMBINED_CORPUS_FILE

    assert CORPUS_REGISTRY["statpearls"].default_path == COMBINED_CORPUS_FILE


def test_corpus_registry_missing_source_is_explicit(tmp_path: Path) -> None:
    from app.rag.data.corpus_registry import combine_registered_corpora

    with pytest.raises(FileNotFoundError, match="Corpus source not found"):
        combine_registered_corpora(
            source_files={"textbooks": str(tmp_path / "missing.json")},
            selected_sources=["textbooks"],
        )


def test_index_metadata_records_phase1_reproducibility_fields(tmp_path: Path) -> None:
    from app.rag.data.medical_corpus.build_vector_index import (
        IndexBuildConfig,
        save_build_metadata,
    )

    config = IndexBuildConfig(
        corpus_file=tmp_path / "phase1_corpus.json",
        index_dir=tmp_path / "phase1_index",
        embedding_model="test-embedding",
        corpus_version="phase1-smoke:test",
        faiss_index_type="FlatIP",
    )
    metadata = save_build_metadata(
        documents=[SimpleNamespace(metadata={"source": "statpearls"})],
        elapsed=1.25,
        config=config,
    )

    assert metadata["embedding_model"] == "test-embedding"
    assert metadata["faiss_index_type"] == "FlatIP"
    assert metadata["corpus_version"] == "phase1-smoke:test"
    assert metadata["sources"] == {"statpearls": 1}
    assert config.build_metadata_file.exists()


def test_phase1_runtime_metadata_distinguishes_embedding_and_llm_api() -> None:
    from app.rag.experiments import run_phase1_ablation as module
    from app.rag.evaluation.eval_shared import EvaluationLLMConfig

    llm_config = EvaluationLLMConfig(
        provider="test-provider",
        model="test-model",
        base_url="https://example.test/v1",
        api_key="secret-key",
    )

    embedding_runtime = module.describe_embedding_runtime()
    llm_runtime = module.describe_llm_runtime(llm_config)

    assert embedding_runtime["backend"] == "api"
    assert embedding_runtime["adapter"].endswith("OpenAIEmbedding")
    assert embedding_runtime["api_used"] is True
    assert "api_key" not in embedding_runtime
    assert llm_runtime["client"].endswith("OpenAILike")
    assert llm_runtime["api_used"] is True
    assert llm_runtime["base_url"] == "https://example.test/v1"
    assert "api_key" not in llm_runtime


def test_phase1_embedding_api_validation_reports_missing_config(monkeypatch) -> None:
    from app.rag.experiments import run_phase1_ablation as module

    monkeypatch.setattr(module, "EMBEDDING_BACKEND", "api")
    monkeypatch.setattr(module, "EMBEDDING_API_BASE_URL", "")
    monkeypatch.setattr(module, "EMBEDDING_API_KEY", "")

    with pytest.raises(ValueError, match="RAG_EMBEDDING_API_BASE_URL"):
        module.validate_phase1_embedding_runtime()


def test_vector_store_uses_official_openai_embedding(monkeypatch) -> None:
    from app.rag.retriever import vector_store as module

    calls = {}

    class FakeEmbedding:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(module, "OpenAIEmbedding", FakeEmbedding)

    module.MedicalVectorStore(
        embedding_model_name="BAAI/bge-m3",
        embedding_api_base_url="https://api.siliconflow.cn/v1",
        embedding_api_key="secret",
        batch_size=8,
    )

    assert calls["model_name"] == "BAAI/bge-m3"
    assert calls["api_base"] == "https://api.siliconflow.cn/v1"
    assert calls["api_key"] == "secret"
    assert calls["embed_batch_size"] == 8


def test_reranker_uses_official_siliconflow_postprocessor(monkeypatch) -> None:
    from app.rag.retriever import reranker as module

    calls = {}

    class FakeSiliconFlowRerank:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(module, "SiliconFlowRerank", FakeSiliconFlowRerank)

    reranker = module.RerankerPipeline(
        cross_encoder_model="BAAI/bge-reranker-v2-m3",
        top_k=2,
        api_url="https://api.siliconflow.cn/v1/rerank",
        api_key="secret",
    )

    assert reranker.cross_encoder.available is True
    assert calls["model"] == "BAAI/bge-reranker-v2-m3"
    assert calls["base_url"] == "https://api.siliconflow.cn/v1/rerank"
    assert calls["api_key"] == "secret"
    assert calls["top_n"] == 2
    assert calls["return_documents"] is False


def test_enhanced_config_splits_retrieval_and_reranker_counts(monkeypatch) -> None:
    from app.rag.evaluation import enhanced_rag_eval as module

    calls = {}

    class FakeHybrid:
        fusion_retriever = "hybrid-retriever"

    class FakeReranker:
        def __init__(
            self,
            *,
            use_cross_encoder,
            cross_encoder_model,
            top_k,
            **kwargs,
        ):
            calls["reranker"] = {
                "use_cross_encoder": use_cross_encoder,
                "model": cross_encoder_model,
                "top_k": top_k,
                "api_url": kwargs["api_url"],
            }
            self.cross_encoder = SimpleNamespace(available=True, model="postprocessor")

    def fake_from_vector_store(cls, vectorstore, **kwargs):
        calls["hybrid"] = kwargs
        return FakeHybrid()

    def fake_query_engine_from_args(**kwargs):
        calls["query_engine"] = kwargs
        return "query-engine"

    monkeypatch.setattr(module, "create_llm", lambda config: "llm")
    monkeypatch.setattr(
        module.HybridRetriever,
        "from_vector_store",
        classmethod(fake_from_vector_store),
    )
    monkeypatch.setattr(module, "RerankerPipeline", FakeReranker)
    monkeypatch.setattr(
        module.RetrieverQueryEngine,
        "from_args",
        staticmethod(fake_query_engine_from_args),
    )

    config = module.EnhancedEvaluationConfig(
        top_k=5,
        retrieval_top_k=20,
        reranker_top_k=5,
        hybrid_alpha=0.25,
        reranker_api_url="https://rerank.example.test/v1/rerank",
        reranker_api_key="secret",
    )
    query_engine = module.build_enhanced_query_engine(SimpleNamespace(), config)

    assert query_engine == "query-engine"
    assert calls["hybrid"]["similarity_top_k"] == 20
    assert calls["hybrid"]["retriever_weights"] == (0.25, 0.75)
    assert calls["reranker"]["top_k"] == 5
    assert calls["reranker"]["api_url"] == "https://rerank.example.test/v1/rerank"
    assert calls["query_engine"]["node_postprocessors"] == ["postprocessor"]
