"""Tests for formal ablation executor resolution and dispatch behavior."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.resolve()))


def test_resolve_stage2_rows_use_stage1_embedding_winners() -> None:
    from app.rag.experiments.formal_ablation_executor import (
        FormalSelectionState,
        is_run_resolved,
        resolve_stage_runs,
    )
    from app.rag.experiments.phase1_formal_ablation import build_formal_matrix

    rows = resolve_stage_runs(
        "2_k_screening",
        build_formal_matrix(),
        FormalSelectionState(
            stage1_top_embeddings=["BAAI/bge-m3", "ncbi/MedCPT"],
        ),
    )

    assert all(is_run_resolved(row) for row in rows)
    assert {row.embedding_model for row in rows[:3]} == {"BAAI/bge-m3"}
    assert {row.embedding_model for row in rows[3:]} == {"ncbi/MedCPT"}
    assert {row.k for row in rows} == {3, 5, 10}


def test_build_eval_configs_use_dev_split_and_formal_metadata(tmp_path: Path) -> None:
    from app.rag.data.data_paths import MEDQA_USMLE_DEV_FILE
    from app.rag.experiments.formal_ablation_executor import (
        FORMAL_GENERATOR_MAX_CONCURRENT,
        FormalExecutionConfig,
        build_enhanced_eval_config,
        build_naive_eval_config,
    )
    from app.rag.experiments.phase1_formal_ablation import build_formal_matrix

    matrix = build_formal_matrix()
    naive_row = next(row for row in matrix if row.run_id == "stage1_naive_bge_m3")
    enhanced_row = next(row for row in matrix if row.run_id == "stage0_advanced_statpearls")
    config = FormalExecutionConfig(max_questions=2)

    naive_config = build_naive_eval_config(naive_row, tmp_path / "index", config)
    enhanced_config = build_enhanced_eval_config(enhanced_row, tmp_path / "index", config)

    assert FORMAL_GENERATOR_MAX_CONCURRENT == 6
    assert naive_config.question_file == MEDQA_USMLE_DEV_FILE
    assert naive_config.dev_size == 0
    assert naive_config.test_size == 2
    assert naive_config.manual_top_k == naive_row.k
    assert naive_config.concurrency.max_concurrent == FORMAL_GENERATOR_MAX_CONCURRENT
    assert naive_config.formal_run_id == naive_row.run_id
    assert naive_config.formal_metadata["query_cache_id"] == (
        "stage1_naive_bge_m3__baai_bge-m3"
    )

    assert enhanced_config.question_file == MEDQA_USMLE_DEV_FILE
    assert enhanced_config.dev_size == 0
    assert enhanced_config.test_size == 2
    assert enhanced_config.top_k == enhanced_row.k
    assert enhanced_config.retrieval_top_k == enhanced_row.reranker_input_count
    assert enhanced_config.reranker_top_k == enhanced_row.reranker_output_count
    assert enhanced_config.use_query_rewrite is True
    assert enhanced_config.concurrency.max_concurrent == FORMAL_GENERATOR_MAX_CONCURRENT
    assert enhanced_config.formal_run_id == enhanced_row.run_id


def test_execute_formal_run_skips_completed_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.rag.data.json_utils import save_json_atomic
    from app.rag.evaluation import formal_artifacts
    from app.rag.experiments import formal_ablation_executor as module
    from app.rag.experiments.phase1_formal_ablation import build_formal_matrix

    monkeypatch.setattr(formal_artifacts, "RUNS_DIR", tmp_path / "runs")
    row = next(row for row in build_formal_matrix() if row.run_id == "stage1_naive_bge_m3")
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
