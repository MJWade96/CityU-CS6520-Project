"""Formal ablation matrix and manifest schema contracts."""

from __future__ import annotations


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


def test_formal_matrix_uses_typed_values_for_resolved_rows() -> None:
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


def test_cache_manifest_declares_reusable_artifact_contracts() -> None:
    from app.rag.experiments import phase1_formal_ablation as module

    rows = module.build_formal_matrix()
    manifest = module.build_cache_manifest(rows)
    required_run_artifacts = {"query_texts", "final_prompts", "llm_outputs"}

    assert {"indexes", "retrieval_cache", "rerank_cache", "runs"}.issubset(
        manifest["base_dirs"]
    )
    assert manifest["cache_top_k"] >= max(module.K_VALUES)
    for row in rows:
        run_cache = manifest["runs"][row.run_id]
        assert required_run_artifacts.issubset(run_cache)
        assert run_cache["query_texts"].endswith("query_texts.jsonl")
        assert run_cache["final_prompts"].endswith("final_prompts.jsonl")
        assert run_cache["llm_outputs"].endswith("llm_outputs.jsonl")
        if row.pipeline == "naive_rag":
            assert run_cache["retrieval_top10"].endswith("retrieval_top10.jsonl")
        else:
            assert run_cache["rerank_outputs"].endswith("rerank_outputs.jsonl")


def test_formal_manifest_uses_current_medqa_splits(monkeypatch) -> None:
    from app.rag.experiments import phase1_formal_ablation as module

    monkeypatch.setattr(module, "load_medqa_usmle_counts", lambda: {"dev": 2, "test": 3})

    manifest = module.build_formal_ablation_manifest()

    assert manifest["dev_split"]["question_count"] == 2
    assert manifest["test_split"]["question_count"] == 3
    assert manifest["legacy_medqa_file_not_used"].endswith("medqa.json")


def test_generator_metadata_is_consistent_between_rows_and_manifest() -> None:
    from app.rag.experiments.phase1_formal_ablation import (
        build_formal_ablation_manifest,
        build_formal_matrix,
    )

    rows = build_formal_matrix()
    manifest = build_formal_ablation_manifest()
    manifest_rows = {row["run_id"]: row for row in manifest["matrix"]}

    assert rows
    assert all(row.generator_model for row in rows)
    assert {row.run_id: row.generator_model for row in rows} == {
        run_id: row["generator_model"] for run_id, row in manifest_rows.items()
    }
