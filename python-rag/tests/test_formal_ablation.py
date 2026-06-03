"""Tests for the formal phase-1 ablation framework."""

from __future__ import annotations

import asyncio
import inspect
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


def test_formal_matrix_rows_record_required_run_metadata() -> None:
    from app.rag.experiments import phase1_formal_ablation as module

    rows = module.build_formal_matrix()
    provider_models = {provider.model for provider in module.EMBEDDING_PROVIDERS}
    resolved_naive_embeddings = {
        row.embedding_model
        for row in rows
        if row.pipeline == "naive_rag"
        and not row.selection_rule
        and row.embedding_model in provider_models
    }

    assert rows
    assert provider_models.issubset(resolved_naive_embeddings)
    for row in rows:
        assert row.stage
        assert row.run_id
        assert row.pipeline in {"naive_rag", "advanced_rag"}
        assert row.corpus_version
        assert row.faiss_index_type
        assert row.generator_model
        assert row.prompt_version
        assert row.dataset_split
        assert isinstance(row.random_seed, int)


def test_cache_manifest_covers_recommendation_cache_items() -> None:
    from app.rag.experiments import phase1_formal_ablation as module

    rows = module.build_formal_matrix()
    manifest = module.build_cache_manifest(rows)

    assert {"indexes", "retrieval_cache", "rerank_cache", "runs"}.issubset(
        manifest["base_dirs"]
    )
    assert manifest["cache_top_k"] >= max(module.K_VALUES)
    for row in rows:
        run_cache = manifest["runs"][row.run_id]
        assert run_cache["chunk_embeddings"].endswith("chunk_embeddings.npy")
        assert run_cache["query_embeddings"].endswith("query_embeddings.npy")
        assert run_cache["faiss_index"].endswith("faiss.index")
        assert run_cache["final_prompts"].endswith("final_prompts.jsonl")
        assert run_cache["llm_outputs"].endswith("llm_outputs.jsonl")
        if row.pipeline == "naive_rag":
            assert run_cache["retrieval_top10"].endswith("retrieval_top10.jsonl")
        else:
            assert run_cache["query_rewrite_outputs"].endswith(
                "query_rewrite_outputs.jsonl"
            )
            assert run_cache["rerank_outputs"].endswith("rerank_outputs.jsonl")


def test_formal_framework_does_not_use_legacy_medqa(monkeypatch) -> None:
    from app.rag.experiments import phase1_formal_ablation as module

    monkeypatch.setattr(module, "load_medqa_usmle_counts", lambda: {"dev": 2, "test": 3})

    manifest = module.build_formal_ablation_manifest()

    assert manifest["dev_split"]["question_count"] == 2
    assert manifest["test_split"]["question_count"] == 3
    assert manifest["legacy_medqa_file_not_used"].endswith("medqa.json")


def test_local_embeddings_stay_in_experiment_framework_not_primary_retriever() -> None:
    from app.rag.experiments import phase1_formal_ablation as module
    from app.rag.retriever.vector_store import MedicalVectorStore

    assert any(provider.name == "medcpt" for provider in module.EMBEDDING_PROVIDERS)
    assert any(
        provider.backend == "local_hf_embedding"
        for provider in module.EMBEDDING_PROVIDERS
    )
    assert "embedding_backend" not in inspect.signature(MedicalVectorStore).parameters
    assert (
        PROJECT_ROOT / "app" / "rag" / "evaluation" / "formal_local_embedding_adapter.py"
    ).exists()


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


def test_duplicate_formal_runtime_is_not_a_supported_surface() -> None:
    assert not (
        PROJECT_ROOT / "app" / "rag" / "experiments" / "formal_ablation_runtime.py"
    ).exists()


def test_medcpt_autodl_script_reuses_medscore_core_without_cli_args() -> None:
    from app.rag.experiments import run_medcpt_embedding_autodl as module

    formatted = module._format_medcpt_article_inputs(
        [{"title": "Title", "content": "Content"}]
    )

    assert module.EMBEDDING_INPUT_FORMAT == "title_content_pair"
    assert module.EMBEDDING_BACKEND == "local_medcpt"
    assert formatted == [["Title", "Content"]]
    assert inspect.signature(module.main).parameters == {}


def test_medcpt_query_autodl_script_embeds_query_texts_only() -> None:
    from app.rag.experiments import run_medcpt_query_embedding_autodl as module

    assert module.MEDCPT_QUERY_MODEL
    assert module.QUERY_INPUT_FORMAT == "retrieval_query_text_only"
    assert inspect.signature(module.main).parameters == {}
    assert {
        spec.pipeline for spec in module.QUERY_EMBEDDING_SPECS
    } == {"naive_rag", "advanced_rag"}


def test_query_rewrite_cache_script_selects_only_advanced_specs() -> None:
    from app.rag.experiments import run_query_rewrite_cache_autodl as module

    specs = module._selected_rewrite_specs()

    assert specs
    assert {spec.pipeline for spec in specs} == {"advanced_rag"}
    assert inspect.signature(module.main).parameters == {}


def test_medcpt_query_autodl_defaults_to_naive_and_advanced_caches() -> None:
    from app.rag.experiments import run_medcpt_query_embedding_autodl as module

    specs_by_pipeline = {spec.pipeline: spec for spec in module.QUERY_EMBEDDING_SPECS}

    assert set(specs_by_pipeline) == {"naive_rag", "advanced_rag"}
    assert specs_by_pipeline["naive_rag"].query_text_source == (
        "medqa_usmle_question_field"
    )
    assert specs_by_pipeline["advanced_rag"].query_text_source == (
        "query_rewrite_pipeline"
    )


def test_local_bge_autodl_scripts_have_no_cli_and_cover_formal_specs() -> None:
    from app.rag.experiments import phase1_formal_ablation as formal_module
    from app.rag.experiments import run_local_bge_embedding_autodl as corpus_module
    from app.rag.experiments import run_local_bge_query_embedding_autodl as module

    specs = {spec.cache_id: spec.pipeline for spec in module.BGE_QUERY_EMBEDDING_SPECS}
    local_models = [
        provider.model
        for provider in formal_module.EMBEDDING_PROVIDERS
        if provider.backend == module.EMBEDDING_BACKEND
    ]
    expected_specs = {}
    for row in formal_module.build_formal_matrix():
        if row.embedding_backend == module.EMBEDDING_BACKEND and row.embedding_model:
            models = [row.embedding_model]
        elif row.selection_rule:
            models = local_models
        else:
            continue
        for embedding_model in models:
            source = (
                "query_rewrite_pipeline"
                if row.pipeline == "advanced_rag"
                else "medqa_usmle_question_field"
            )
            expected_specs[module.query_cache_id(row.run_id, embedding_model)] = (
                row.pipeline,
                source,
            )

    assert corpus_module.EMBEDDING_BACKEND == "local_hf_embedding"
    assert corpus_module.CORPUS_BATCH_SIZE == 128
    assert module.QUERY_BATCH_SIZE == 256
    assert inspect.signature(corpus_module.main).parameters == {}
    assert inspect.signature(module.main).parameters == {}
    assert specs == {
        cache_id: pipeline for cache_id, (pipeline, _source) in expected_specs.items()
    }
    assert {
        spec.cache_id: spec.query_text_source
        for spec in module.BGE_QUERY_EMBEDDING_SPECS
    } == {
        cache_id: source for cache_id, (_pipeline, source) in expected_specs.items()
    }


def test_local_rerank_cache_autodl_script_has_no_cli_and_uses_llamaindex() -> None:
    from app.rag.evaluation.formal_local_rerank_cache import LOCAL_RERANKER_BACKEND
    from app.rag.experiments import run_local_rerank_cache_autodl as module

    class FakeReranker:
        def postprocess_nodes(self, nodes, query_str):
            assert query_str == "query"
            return nodes[:1]

    rows = module.rerank_cache_rows(
        "cache-a",
        [
            {
                "question_id": "dev-1",
                "query_text": "query",
                "candidates": [{"text": "context", "score": 0.5}],
            }
        ],
        FakeReranker(),
    )

    assert module.FUSION_CANDIDATES_FILENAME == "fusion_candidates.jsonl"
    assert inspect.signature(module.main).parameters == {}
    assert rows[0]["reranker_backend"] == LOCAL_RERANKER_BACKEND
    assert rows[0]["reranked_candidates"][0]["text"] == "context"


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
