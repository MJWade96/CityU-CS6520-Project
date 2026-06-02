"""Tests for the formal phase-1 ablation framework."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.resolve()))


def test_medqa_usmle_adapter_loads_dev_and_test_shapes(tmp_path: Path) -> None:
    from app.rag.data.benchmarks.medqa_usmle import load_medqa_usmle_jsonl
    from app.rag.evaluation.eval_shared import load_questions

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
    shared_records = load_questions(str(split_file))

    assert records[0]["id"] == "dev-1"
    assert records[0]["options"] == ["Alpha", "Beta"]
    assert records[0]["answer_index"] == 1
    assert records[0]["answer_idx"] == "B"
    assert records[0]["split"] == "dev"
    assert shared_records == records


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
    bge_rows = [row for row in rows if row.embedding_model == "BAAI/bge-m3"]
    assert {row.embedding_backend for row in bge_rows} == {"local_hf_embedding"}
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

    rows = module.build_formal_matrix()
    manifest = module.build_cache_manifest(rows)

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

    medcpt_run = next(row for row in rows if row.run_id == "stage1_naive_medcpt")
    medcpt_manifest = manifest["runs"][medcpt_run.run_id]
    assert medcpt_manifest["chunk_embeddings"].endswith("chunk_embeddings.npy")
    assert medcpt_manifest["faiss_index"].endswith("faiss.index")
    assert medcpt_manifest["query_embeddings"].endswith("query_embeddings.npy")
    assert medcpt_manifest["retrieval_top80"].endswith("retrieval_top80.jsonl")


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


def test_local_embeddings_stay_in_experiment_framework_not_primary_retriever() -> None:
    from app.rag.experiments import phase1_formal_ablation as module

    vector_store_source = (
        PROJECT_ROOT / "app" / "rag" / "retriever" / "vector_store.py"
    ).read_text(encoding="utf-8")

    assert any(provider.name == "medcpt" for provider in module.EMBEDDING_PROVIDERS)
    assert any(
        provider.backend == "local_hf_embedding"
        for provider in module.EMBEDDING_PROVIDERS
        if provider.model.startswith("BAAI/")
    )
    assert not (
        PROJECT_ROOT / "app" / "rag" / "experiments" / "formal_ablation_runtime.py"
    ).exists()
    assert "MedCPT" not in vector_store_source
    assert "local_medcpt" not in vector_store_source
    assert "local_hf_embedding" not in vector_store_source


def test_formal_matrix_uses_typed_values_not_runtime_string_parameters() -> None:
    from app.rag.experiments.phase1_formal_ablation import build_formal_matrix

    rows = build_formal_matrix()
    resolved = [row for row in rows if not row.selection_rule]
    unresolved = [row for row in rows if row.selection_rule]

    assert all(isinstance(row.k, int) for row in resolved)
    assert all(
        row.reranker_input_count is None or isinstance(row.reranker_input_count, int)
        for row in resolved
    )
    assert all(row.embedding_model is None for row in unresolved)
    assert not (
        PROJECT_ROOT / "app" / "rag" / "experiments" / "formal_ablation_runtime.py"
    ).exists()


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
    assert "QueryRewritePipeline" not in source
    assert "build_formal_matrix" not in source
    assert "select_medcpt_runs" not in source
    assert "build_query" not in source
    assert "build_medical_eval_prompt" not in source
    assert "retrieve_top80" not in source
    assert "hybrid_retrieve_top80" not in source
    assert "rerank_rows" not in source
    assert "faiss" not in source
    assert "argparse" not in source
    assert "parse_args" not in source


def test_query_rewrite_cache_script_only_rewrites_queries() -> None:
    source = (
        PROJECT_ROOT
        / "app"
        / "rag"
        / "experiments"
        / "run_query_rewrite_cache_autodl.py"
    ).read_text(encoding="utf-8")

    assert "QueryRewritePipeline" in source
    assert "MEDCPT_QUERY_MODEL" not in source
    assert "AutoModel" not in source
    assert "embed_query_texts" not in source
    assert "np.save" not in source
    assert "build_query" not in source
    assert "build_medical_eval_prompt" not in source
    assert "retrieve_top80" not in source
    assert "hybrid_retrieve_top80" not in source
    assert "rerank_rows" not in source
    assert "faiss" not in source
    assert "argparse" not in source
    assert "parse_args" not in source


def test_medcpt_query_autodl_defaults_to_naive_and_advanced_caches() -> None:
    from app.rag.experiments import run_medcpt_query_embedding_autodl as module

    specs = {spec.cache_id: spec.pipeline for spec in module.QUERY_EMBEDDING_SPECS}

    assert specs == {
        "stage1_naive_medcpt": "naive_rag",
        "advanced_medcpt_rewritten_query": "advanced_rag",
    }


def test_local_bge_autodl_scripts_have_no_cli_and_cover_formal_specs() -> None:
    from app.rag.experiments import run_local_bge_query_embedding_autodl as module

    corpus_source = (
        PROJECT_ROOT
        / "app"
        / "rag"
        / "experiments"
        / "run_local_bge_embedding_autodl.py"
    ).read_text(encoding="utf-8")
    query_source = (
        PROJECT_ROOT
        / "app"
        / "rag"
        / "experiments"
        / "run_local_bge_query_embedding_autodl.py"
    ).read_text(encoding="utf-8")
    specs = {spec.cache_id: spec.pipeline for spec in module.BGE_QUERY_EMBEDDING_SPECS}

    assert "HuggingFaceEmbedding" in corpus_source
    assert "HuggingFaceEmbedding" in query_source
    assert "CORPUS_BATCH_SIZE = 8" in corpus_source
    assert "QUERY_BATCH_SIZE = 256" in query_source
    assert "argparse" not in corpus_source
    assert "parse_args" not in corpus_source
    assert "argparse" not in query_source
    assert "parse_args" not in query_source
    assert "stage1_naive_bge_m3__baai_bge-m3" in specs
    assert "stage1_naive_bge_large_en_v1_5__baai_bge-large-en-v1p5" in specs
    assert "stage3_advanced_stage2_top1_embedding_k__baai_bge-m3" in specs
    assert specs["stage3_advanced_stage2_top1_embedding_k__baai_bge-m3"] == "advanced_rag"


def test_local_bge_embedding_batches_long_corpus_texts() -> None:
    from app.rag.experiments import run_local_bge_embedding_autodl as module

    class FakeEmbedding:
        def __init__(self):
            self.batch_sizes = []

        def get_text_embedding_batch(self, texts, show_progress):
            self.batch_sizes.append(len(texts))
            return [[1.0, 0.0] for _ in texts]

    fake_embedding = FakeEmbedding()
    embeddings = module.embed_texts(
        fake_embedding,
        ["a", "b", "c", "d", "e"],
        batch_size=2,
        progress_label="test",
    )

    assert fake_embedding.batch_sizes == [2, 2, 1]
    assert embeddings.shape == (5, 2)
    assert np.allclose(embeddings[:, 0], 1.0)


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
        if spec.cache_id == "advanced_medcpt_rewritten_query"
    )
    monkeypatch.setattr(module, "_query_texts_path", lambda _: tmp_path / "missing.jsonl")

    with pytest.raises(FileNotFoundError) as exc_info:
        module.resolve_query_text_rows(spec, [{"id": "dev-1", "question": "Question?"}])

    assert "run_query_rewrite_cache_autodl.py" in str(exc_info.value)


def test_medcpt_advanced_query_text_rows_use_rewritten_query(
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
        spec
        for spec in QUERY_EMBEDDING_SPECS
        if spec.cache_id == "advanced_medcpt_rewritten_query"
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
    checkpoint_rows = [
        json.loads(line) for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
    ]
    error_rows = [
        json.loads(line) for line in errors_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["question_id"] for row in checkpoint_rows] == ["dev-1"]
    assert error_rows[-1]["question_id"] == "dev-2"
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

    final_rows = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["question_id"] for row in final_rows] == ["dev-1", "dev-2"]
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

    rows_a = [
        json.loads(line) for line in output_paths["advanced_a"].read_text(encoding="utf-8").splitlines()
    ]
    rows_b = [
        json.loads(line) for line in output_paths["advanced_b"].read_text(encoding="utf-8").splitlines()
    ]
    assert rewrite_calls["count"] == 1
    assert rows_a == rows_b
