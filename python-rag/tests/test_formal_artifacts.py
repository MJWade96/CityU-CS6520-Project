"""Tests for formal ablation artifact persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.resolve()))


def _metadata():
    from app.rag.evaluation.formal_artifacts import FormalRunMetadata

    return FormalRunMetadata(
        run_id="stage1_naive_test",
        stage="1_embedding_screening",
        pipeline="naive_rag",
        corpus_version="statpearls_textbooks",
        embedding_model="BAAI/bge-m3",
        embedding_backend="siliconflow_api",
        faiss_index_type="FlatIP",
        k=5,
        alpha=None,
        reranker_input_count=0,
        reranker_output_count=0,
        query_enhancement_setting="off",
        generator_model="Qwen3-8B",
        prompt_version="medical_mcq_v1",
        dataset_split="dev",
        random_seed=6520,
        cache_ids={
            "retrieval": "stage1_naive_test__retrieval",
            "rerank": "stage1_naive_test__rerank",
        },
    )


def test_formal_artifact_recorder_finalizes_group_manifests(tmp_path: Path) -> None:
    from app.rag.evaluation.formal_artifacts import FormalArtifactRecorder

    recorder = FormalArtifactRecorder(_metadata(), results_dir=tmp_path)
    recorder.start_run(expected_question_count=1)
    recorder.write_query_text(
        {
            "question_id": "dev-1",
            "question": "Question?",
            "query_text": "Question?",
        }
    )
    recorder.write_retrieval(
        "retrieval_top10",
        {
            "question_id": "dev-1",
            "query_text": "Question?",
            "candidates": [],
        },
    )
    recorder.write_selected_contexts({"question_id": "dev-1", "contexts": []})
    recorder.write_final_prompt({"question_id": "dev-1", "prompt": "Prompt"})
    recorder.write_llm_output({"question_id": "dev-1", "content": "Answer: A"})
    recorder.write_evaluation_output({"question_id": "dev-1", "is_correct": True})
    recorder.save_run_checkpoint({"processed_questions": 1})
    recorder.finalize_run(
        {
            "total_questions": 1,
            "processed_questions": 1,
            "correct": 1,
            "accuracy": 1.0,
        }
    )

    run_dir = tmp_path / "runs" / "stage1_naive_test"
    retrieval_dir = tmp_path / "retrieval_cache" / "stage1_naive_test__retrieval"

    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (retrieval_dir / "manifest.json").exists()
    assert not (run_dir / "run_checkpoint.json").exists()
    assert not (run_dir / "evaluation_outputs.checkpoint.jsonl").exists()

    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    retrieval_manifest = json.loads(
        (retrieval_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["status"] == "completed"
    assert run_manifest["row_count"] == 1
    assert retrieval_manifest["artifact_group"] == "retrieval"
    assert retrieval_manifest["status"] == "completed"


def test_make_recorder_requires_matching_metadata() -> None:
    import pytest

    from app.rag.evaluation.formal_artifacts import make_recorder

    payload = _metadata().__dict__.copy()
    payload["run_id"] = "different"

    with pytest.raises(ValueError, match="does not match"):
        make_recorder("stage1_naive_test", payload)
