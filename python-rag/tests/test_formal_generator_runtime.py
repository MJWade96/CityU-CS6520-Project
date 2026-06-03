"""Tests for formal generator metadata contracts."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.resolve()))


def test_formal_matrix_records_generator_model_for_each_run() -> None:
    from app.rag.experiments.phase1_formal_ablation import (
        build_formal_ablation_manifest,
        build_formal_matrix,
    )

    rows = build_formal_matrix()
    manifest = build_formal_ablation_manifest()
    manifest_rows = {row["run_id"]: row for row in manifest["matrix"]}

    assert rows
    assert all(row.generator_model for row in rows)
    assert {
        row.run_id: row.generator_model for row in rows
    } == {
        run_id: row["generator_model"] for run_id, row in manifest_rows.items()
    }
