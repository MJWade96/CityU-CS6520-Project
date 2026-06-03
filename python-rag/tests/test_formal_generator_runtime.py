"""Tests for formal generator defaults."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.resolve()))


def test_formal_matrix_records_qwen3_4b_generator() -> None:
    from app.rag.experiments.phase1_formal_ablation import build_formal_matrix

    rows = build_formal_matrix()

    assert {row.generator_model for row in rows} == {"Qwen3-4B"}
