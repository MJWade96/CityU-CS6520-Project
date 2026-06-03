"""Formal ablation manifest artifact consumability."""

from __future__ import annotations


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
