"""Tests for formal generator defaults and concurrent final-answer calls."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.resolve()))


def test_formal_matrix_records_qwen3_8b_generator() -> None:
    from app.rag.experiments.phase1_formal_ablation import build_formal_matrix

    rows = build_formal_matrix()

    assert {row.generator_model for row in rows} == {"Qwen3-8B"}


def test_formal_final_answers_run_concurrently(monkeypatch) -> None:
    from app.rag.evaluation.eval_shared import EvaluationLLMConfig
    from app.rag.experiments import formal_ablation_runtime as runtime
    from app.rag.experiments.phase1_formal_ablation import build_formal_matrix

    active = 0
    max_active = 0

    async def fake_call_llm(ctx, prompt):
        nonlocal active, max_active
        async with ctx.semaphore:
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
        return "Answer: A"

    monkeypatch.setenv("RAG_EVAL_MAX_CONCURRENT", "2")
    monkeypatch.setattr(runtime, "call_llm", fake_call_llm)

    run = next(row for row in build_formal_matrix() if row.run_id == "stage1_naive_bge_m3")
    questions = [
        {"id": "dev-1", "question": "Q1?", "options": ["A"], "answer_index": 0},
        {"id": "dev-2", "question": "Q2?", "options": ["A"], "answer_index": 0},
        {"id": "dev-3", "question": "Q3?", "options": ["A"], "answer_index": 0},
    ]
    contexts = [
        [{"score": 1.0, "text": "context"}],
        [{"score": 1.0, "text": "context"}],
        [{"score": 1.0, "text": "context"}],
    ]

    generation = asyncio.run(
        runtime.evaluate_final_answers(
            run=run,
            selected_questions=questions,
            contexts_by_question=contexts,
            llm=EvaluationLLMConfig(api_key="test-key"),
        )
    )

    assert max_active == 2
    assert generation["correct"] == 3
    assert generation["max_concurrent"] == 2
