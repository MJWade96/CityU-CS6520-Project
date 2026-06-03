"""Tests for formal ablation executor resolution and dispatch behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path


def _formal_row(
    *,
    stage: str = "stage",
    run_id: str = "run",
    pipeline: str = "naive_rag",
    embedding_model: str | None = "provider-a",
    embedding_backend: str | None = "local_hf_embedding",
    k: int | None = 5,
    alpha: float | None = None,
    reranker_input_count: int | None = 0,
    reranker_output_count: int | None = 0,
    selection_rule: str = "",
):
    from app.rag.experiments.phase1_formal_ablation import FormalRunSpec

    return FormalRunSpec(
        stage=stage,
        run_id=run_id,
        pipeline=pipeline,
        corpus_version="corpus-a",
        embedding_model=embedding_model,
        embedding_backend=embedding_backend,
        faiss_index_type="FlatIP",
        k=k,
        alpha=alpha,
        reranker_input_count=reranker_input_count,
        reranker_output_count=reranker_output_count,
        query_enhancement_setting="on" if pipeline == "advanced_rag" else "off",
        generator_model="generator-a",
        prompt_version="prompt-a",
        dataset_split="split-a",
        random_seed=1,
        selection_rule=selection_rule,
    )


def test_resolve_ranked_rows_use_prior_embedding_winners(monkeypatch) -> None:
    from app.rag.experiments import formal_ablation_executor as module

    stage = module.DEFAULT_RUN_STAGES[2]
    monkeypatch.setattr(
        module,
        "_provider_for_model",
        lambda model: (model, f"backend-for-{model}"),
    )
    rows = module.resolve_stage_runs(
        stage,
        [
            _formal_row(
                stage=stage,
                run_id="candidate_top1_k3",
                embedding_model=None,
                embedding_backend=None,
                k=3,
                selection_rule="use prior winner 1",
            ),
            _formal_row(
                stage=stage,
                run_id="candidate_top2_k7",
                embedding_model=None,
                embedding_backend=None,
                k=7,
                selection_rule="use prior winner 2",
            ),
        ],
        module.FormalSelectionState(stage1_top_embeddings=["winner-a", "winner-b"]),
    )

    assert all(module.is_run_resolved(row) for row in rows)
    assert [row.embedding_model for row in rows] == ["winner-a", "winner-b"]
    assert [row.embedding_backend for row in rows] == [
        "backend-for-winner-a",
        "backend-for-winner-b",
    ]
    assert [row.k for row in rows] == [3, 7]


def test_build_eval_configs_preserve_row_values_and_formal_metadata(tmp_path: Path) -> None:
    from app.rag.experiments import formal_ablation_executor as module

    naive_row = _formal_row(run_id="naive-run", k=4)
    enhanced_row = _formal_row(
        run_id="advanced-run",
        pipeline="advanced_rag",
        k=6,
        alpha=0.25,
        reranker_input_count=18,
        reranker_output_count=6,
    )
    config = module.FormalExecutionConfig(max_questions=2)

    naive_config = module.build_naive_eval_config(naive_row, tmp_path / "index", config)
    enhanced_config = module.build_enhanced_eval_config(
        enhanced_row, tmp_path / "index", config
    )

    assert naive_config.dev_size == 0
    assert naive_config.test_size == 2
    assert naive_config.manual_top_k == naive_row.k
    assert naive_config.concurrency.max_concurrent == (
        module.FORMAL_GENERATOR_MAX_CONCURRENT
    )
    assert naive_config.formal_run_id == naive_row.run_id
    assert naive_config.formal_metadata["run_id"] == naive_row.run_id
    assert naive_config.formal_metadata["query_cache_id"] == (
        f"{naive_row.run_id}__provider-a"
    )

    assert enhanced_config.dev_size == 0
    assert enhanced_config.test_size == 2
    assert enhanced_config.top_k == enhanced_row.k
    assert enhanced_config.retrieval_top_k == enhanced_row.reranker_input_count
    assert enhanced_config.reranker_top_k == enhanced_row.reranker_output_count
    assert enhanced_config.use_query_rewrite is True
    assert enhanced_config.concurrency.max_concurrent == (
        module.FORMAL_GENERATOR_MAX_CONCURRENT
    )
    assert enhanced_config.formal_run_id == enhanced_row.run_id
    assert enhanced_config.formal_metadata["run_id"] == enhanced_row.run_id


def test_execute_formal_run_skips_completed_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.rag.data.json_utils import save_json_atomic
    from app.rag.evaluation import formal_artifacts
    from app.rag.experiments import formal_ablation_executor as module

    monkeypatch.setattr(formal_artifacts, "RUNS_DIR", tmp_path / "runs")
    row = _formal_row(run_id="completed-run")
    run_path = formal_artifacts.run_dir(row.run_id)
    save_json_atomic(run_path / "metrics.json", {"run_id": row.run_id, "accuracy": 0.75})
    save_json_atomic(run_path / "manifest.json", {"status": "completed"})
    monkeypatch.setattr(
        module,
        "ensure_formal_index",
        lambda _: (_ for _ in ()).throw(AssertionError("should not materialize")),
    )

    metrics = asyncio.run(module.execute_formal_run(row, module.FormalExecutionConfig()))

    assert metrics == {"run_id": row.run_id, "accuracy": 0.75}
