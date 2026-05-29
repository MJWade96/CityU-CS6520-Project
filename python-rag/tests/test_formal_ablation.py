"""Tests for the formal phase-1 ablation framework."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.resolve()))


def test_medqa_usmle_adapter_loads_dev_and_test_shapes(tmp_path: Path) -> None:
    from app.rag.data.benchmarks.medqa_usmle import load_medqa_usmle_jsonl

    split_file = tmp_path / "dev.jsonl"
    split_file.write_text(
        json.dumps(
            {
                "question": "Question?",
                "answer": "Beta",
                "options": {"A": "Alpha", "B": "Beta"},
                "meta_info": "step1",
                "answer_idx": "B",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_medqa_usmle_jsonl(split_file, split="dev")

    assert records[0]["id"] == "dev-1"
    assert records[0]["options"] == ["Alpha", "Beta"]
    assert records[0]["answer_index"] == 1
    assert records[0]["answer_idx"] == "B"
    assert records[0]["split"] == "dev"


def test_formal_matrix_uses_dev_and_includes_required_embeddings() -> None:
    from app.rag.experiments import phase1_formal_ablation as module

    rows = module.build_formal_matrix()
    embeddings = {row.embedding_model for row in rows}
    splits = {row.dataset_split for row in rows}
    stages = {row.stage for row in rows}

    assert splits == {"dev"}
    assert "BAAI/bge-m3" in embeddings
    assert "BAAI/bge-large-en-v1.5" in embeddings
    assert "ncbi/MedCPT" in embeddings
    assert {
        "0_corpus_ablation",
        "1_embedding_screening",
        "2_k_screening",
        "3_advanced_review",
        "4_alpha_ablation",
        "5_reranker_input_ablation",
    }.issubset(stages)


def test_cache_manifest_covers_recommendation_cache_items() -> None:
    from app.rag.experiments import phase1_formal_ablation as module

    manifest = module.build_cache_manifest(module.build_formal_matrix())

    assert set(module.CACHE_KEYS) == {
        "chunk_embeddings",
        "query_embeddings",
        "faiss_index",
        "retrieval_top80",
        "rerank_outputs",
        "final_prompts",
        "llm_outputs",
        "token_usage",
        "estimated_token_cost",
    }
    assert manifest["cache_top_k"] == 80
    first_run = next(iter(manifest["runs"].values()))
    assert set(module.CACHE_KEYS).issubset(first_run)


def test_formal_framework_does_not_use_legacy_medqa(monkeypatch) -> None:
    from app.rag.experiments import phase1_formal_ablation as module

    monkeypatch.setattr(module, "load_medqa_usmle_counts", lambda: {"dev": 2, "test": 3})

    manifest = module.build_formal_ablation_manifest()

    assert manifest["dev_split"]["question_count"] == 2
    assert manifest["test_split"]["question_count"] == 3
    assert manifest["legacy_medqa_file_not_used"].endswith("medqa.json")
    assert manifest["stage6_faiss_index_ablation"]["status"] == (
        "out_of_scope_for_current_phase"
    )


def test_medcpt_stays_in_experiment_framework_not_primary_retriever() -> None:
    from app.rag.experiments import phase1_formal_ablation as module
    from app.rag.experiments import formal_ablation_runtime as runtime

    vector_store_source = (
        PROJECT_ROOT / "app" / "rag" / "retriever" / "vector_store.py"
    ).read_text(encoding="utf-8")

    assert any(provider.name == "medcpt" for provider in module.EMBEDDING_PROVIDERS)
    assert runtime.MEDCPT_QUERY_MODEL == "ncbi/MedCPT-Query-Encoder"
    assert runtime.MEDCPT_ARTICLE_MODEL == "ncbi/MedCPT-Article-Encoder"
    assert "MedCPT" not in vector_store_source
    assert "local_medcpt" not in vector_store_source


def test_formal_runtime_declares_real_cache_artifact_paths() -> None:
    from app.rag.experiments import formal_ablation_runtime as runtime
    from app.rag.experiments.phase1_formal_ablation import build_formal_matrix

    run = next(row for row in build_formal_matrix() if row.run_id == "stage1_naive_bge_m3")
    run_paths = runtime.formal_run_paths(run)
    index_paths = runtime.dense_index_paths(run)

    assert str(index_paths.chunk_embeddings).endswith("chunk_embeddings.npy")
    assert str(index_paths.faiss_index).endswith("faiss.index")
    assert str(index_paths.bm25_index_dir).endswith("bm25")
    assert str(run_paths.query_embeddings).endswith("query_embeddings.npy")
    assert str(run_paths.retrieval_top80).endswith("retrieval_top80.jsonl")
    assert str(run_paths.final_prompts).endswith("final_prompts.jsonl")
    assert str(run_paths.llm_outputs).endswith("llm_outputs.jsonl")
    assert str(run_paths.token_usage).endswith("token_usage.json")
    assert str(run_paths.estimated_token_cost).endswith("estimated_token_cost.json")


def test_formal_runtime_uses_llamaindex_bm25_not_ad_hoc_rank_bm25() -> None:
    runtime_source = (
        PROJECT_ROOT / "app" / "rag" / "experiments" / "formal_ablation_runtime.py"
    ).read_text(encoding="utf-8")

    assert "BM25Retriever.from_defaults" in runtime_source
    assert "BM25Retriever.from_persist_dir" in runtime_source
    assert "rank_bm25" not in runtime_source
    assert "BM25Okapi" not in runtime_source
    assert "QueryFusionRetriever" in runtime_source


def test_formal_runtime_requires_external_medcpt_embeddings_before_local_index() -> None:
    from app.rag.experiments import formal_ablation_runtime as runtime
    from app.rag.experiments.phase1_formal_ablation import build_formal_matrix

    run = next(row for row in build_formal_matrix() if row.run_id == "stage1_naive_medcpt")

    with pytest.raises(FileNotFoundError) as exc_info:
        runtime.ensure_chunk_embeddings(run, [{"text": "example"}])

    assert "Generate" in str(exc_info.value)


def test_formal_documents_are_minimal_flat_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.rag.experiments import formal_ablation_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "combine_registered_corpora",
        lambda selected_sources: {
            "records": [
                {
                    "id": "doc-1",
                    "title": "Title",
                    "content": "Content",
                    "contents": "Title. Content",
                    "source": "statpearls",
                    "section": "ignored",
                }
            ]
        },
    )

    documents = runtime.load_corpus_documents("statpearls")

    assert documents == [
        {
            "doc_id": "doc-1",
            "title": "Title",
            "source": "statpearls",
            "text": "Title. Content",
        }
    ]


def test_medcpt_autodl_script_reuses_medscore_core_without_cli_args() -> None:
    source = (
        PROJECT_ROOT
        / "app"
        / "rag"
        / "experiments"
        / "run_medcpt_embedding_autodl.py"
    ).read_text(encoding="utf-8")

    assert "class CustomizeSentenceTransformer" in source
    assert 'Pooling(transformer_model.get_word_embedding_dimension(), "cls")' in source
    assert 'EMBEDDING_INPUT_FORMAT = "title_content_pair"' in source
    assert "[[str(record[\"title\"]), str(record[\"content\"])]" in source
    assert "argparse" not in source
    assert "parse_args" not in source


def test_medcpt_query_autodl_script_embeds_query_texts_only() -> None:
    source = (
        PROJECT_ROOT
        / "app"
        / "rag"
        / "experiments"
        / "run_medcpt_query_embedding_autodl.py"
    ).read_text(encoding="utf-8")

    assert 'MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"' in source
    assert 'QUERY_INPUT_FORMAT = "retrieval_query_text_only"' in source
    assert "QueryRewritePipeline" in source
    assert "build_query" not in source
    assert "build_medical_eval_prompt" not in source
    assert "retrieve_top80" not in source
    assert "hybrid_retrieve_top80" not in source
    assert "rerank_rows" not in source
    assert "faiss" not in source
    assert "argparse" not in source
    assert "parse_args" not in source


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


def test_medcpt_advanced_query_text_rows_use_rewritten_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.rag.evaluation.eval_shared import EvaluationLLMConfig
    from app.rag.experiments import run_medcpt_query_embedding_autodl as module

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
