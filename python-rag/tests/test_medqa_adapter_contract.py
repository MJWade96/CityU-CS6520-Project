"""MedQA-USMLE adapter behavior contracts."""

from __future__ import annotations

import json
from pathlib import Path


def test_medqa_usmle_adapter_loads_shared_question_shape(tmp_path: Path) -> None:
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
