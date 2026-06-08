"""Tests for small formal artifact file helpers."""

from __future__ import annotations

import json
from pathlib import Path


def test_jsonl_helpers_append_load_and_report_completed_ids(tmp_path: Path) -> None:
    from app.rag.evaluation.formal_artifacts import (
        append_jsonl,
        append_jsonl_if_question_missing,
        append_jsonl_with_checkpoint,
        checkpoint_path,
        completed_question_ids,
        load_jsonl,
    )

    path = tmp_path / "cache" / "rows.jsonl"

    assert load_jsonl(path) == []
    assert completed_question_ids(path) == set()

    append_jsonl(path, {"question_id": "dev-1", "value": "alpha"})
    append_jsonl(path, {"question_id": "dev-2", "value": "beta"})

    assert load_jsonl(path) == [
        {"question_id": "dev-1", "value": "alpha"},
        {"question_id": "dev-2", "value": "beta"},
    ]
    assert completed_question_ids(path) == {"dev-1", "dev-2"}

    checkpointed_path = tmp_path / "cache" / "checkpointed.jsonl"
    append_jsonl_with_checkpoint(checkpointed_path, {"question_id": "dev-3"})
    assert load_jsonl(checkpointed_path) == [{"question_id": "dev-3"}]
    assert load_jsonl(checkpoint_path(checkpointed_path)) == [{"question_id": "dev-3"}]

    append_jsonl_if_question_missing(checkpointed_path, {"question_id": "dev-3"})
    append_jsonl_if_question_missing(checkpointed_path, {"question_id": "dev-4"})
    assert load_jsonl(checkpointed_path) == [
        {"question_id": "dev-3"},
        {"question_id": "dev-4"},
    ]


def test_successful_evaluations_resolve_historical_generator_errors() -> None:
    from app.rag.evaluation.formal_artifacts import unresolved_generator_error_ids

    errors = {"dev-1": {}, "dev-2": {}}
    evaluations = {"dev-1": {}}

    assert unresolved_generator_error_ids(errors, evaluations) == {"dev-2"}


def test_generator_outputs_keep_reasoning_separate(tmp_path: Path) -> None:
    from app.rag.evaluation.formal_artifacts import (
        append_generator_outputs_if_missing,
        load_jsonl,
    )

    llm_rows = {}
    evaluation_rows = {}
    append_generator_outputs_if_missing(
        llm_outputs_path=tmp_path / "llm_outputs.jsonl",
        evaluation_outputs_path=tmp_path / "evaluation_outputs.jsonl",
        question_id="test-1",
        response="Answer: A",
        reasoning_content="reasoning trace",
        result={"is_correct": True},
        llm_rows=llm_rows,
        evaluation_rows=evaluation_rows,
    )

    assert load_jsonl(tmp_path / "llm_outputs.jsonl") == [
        {
            "question_id": "test-1",
            "response": "Answer: A",
            "reasoning_content": "reasoning trace",
        }
    ]


def test_json_and_path_helpers_use_formal_locations(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.rag.evaluation import formal_artifacts as module

    monkeypatch.setattr(module, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(module, "RETRIEVAL_CACHE_DIR", tmp_path / "retrieval_cache")
    monkeypatch.setattr(module, "RERANK_CACHE_DIR", tmp_path / "rerank_cache")

    module.write_metrics("run-a", {"accuracy": 0.5})
    module.write_run_manifest("run-a", {"status": "completed"})

    assert module.run_dir("run-a") == tmp_path / "runs" / "run-a"
    assert module.retrieval_cache_dir("cache-a") == tmp_path / "retrieval_cache" / "cache-a"
    assert module.rerank_cache_dir("cache-a") == tmp_path / "rerank_cache" / "cache-a"
    assert json.loads((tmp_path / "runs" / "run-a" / "metrics.json").read_text()) == {
        "accuracy": 0.5
    }
    assert json.loads((tmp_path / "runs" / "run-a" / "manifest.json").read_text()) == {
        "status": "completed"
    }
