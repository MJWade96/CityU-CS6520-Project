"""Formal ablation orchestration over the existing evaluation stacks."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.rag.data.data_paths import (
    MEDQA_USMLE_DEV_FILE,
    MEDQA_USMLE_TEST_FILE,
    RUNS_DIR,
    ensure_data_directories,
)
from app.rag.data.json_utils import load_json_safe, save_json_atomic
from app.rag.evaluation.config import NaiveRAGEvalConfig
from app.rag.evaluation.eval_shared import ConcurrencyConfig
from app.rag.evaluation.enhanced_rag_eval import EnhancedEvaluationConfig
from app.rag.evaluation.formal_artifacts import run_dir, write_metrics, write_run_manifest
from app.rag.evaluation.naive_rag_eval import run_complete_evaluation
from app.rag.experiments.formal_index_materializer import ensure_formal_index
from app.rag.experiments.phase1_formal_ablation import (
    BASELINE_RERANKER_MULTIPLIER,
    EMBEDDING_PROVIDERS,
    LOCAL_EMBEDDING_BACKENDS,
    RERANKER_INPUT_MULTIPLIERS,
    RUN_ID,
    FormalRunSpec,
    build_cache_manifest,
    build_formal_matrix,
    _slug,
    write_csv,
)


DEFAULT_RUN_STAGES = (
    "0_corpus_ablation",
    "1_embedding_screening",
    "2_k_screening",
    "3_advanced_review",
    "4_alpha_ablation",
    "5_reranker_input_ablation",
)

FORMAL_GENERATOR_MAX_CONCURRENT = 2


@dataclass(frozen=True)
class FormalExecutionConfig:
    """Script-internal formal execution settings."""

    run_stages: Sequence[str] = DEFAULT_RUN_STAGES
    dataset_split: str = "dev"
    max_questions: Optional[int] = None
    resume: bool = True
    force: bool = False


@dataclass
class FormalSelectionState:
    """Resolved winners carried from one formal stage to the next."""

    stage1_top_embeddings: List[str] = field(default_factory=list)
    stage2_top_pairs: List[Tuple[str, int]] = field(default_factory=list)
    stage3_best_embedding: Optional[str] = None
    stage3_best_k: Optional[int] = None
    stage4_best_alpha: Optional[float] = None
    stage5_best_reranker_input_count: Optional[int] = None


def _stage_root() -> Path:
    return RUNS_DIR / RUN_ID


def _stage_checkpoint_path(stage: str) -> Path:
    return _stage_root() / "stage_checkpoints" / f"{stage}.json"


def _stage_summary_path(stage: str) -> Path:
    return _stage_root() / "stage_summaries" / f"{stage}.json"


def _state_payload(state: FormalSelectionState) -> Dict[str, Any]:
    payload = asdict(state)
    payload["stage2_top_pairs"] = [list(pair) for pair in state.stage2_top_pairs]
    return payload


def _state_from_payload(payload: Dict[str, Any]) -> FormalSelectionState:
    return FormalSelectionState(
        stage1_top_embeddings=list(payload.get("stage1_top_embeddings", [])),
        stage2_top_pairs=[
            (str(pair[0]), int(pair[1])) for pair in payload.get("stage2_top_pairs", [])
        ],
        stage3_best_embedding=payload.get("stage3_best_embedding"),
        stage3_best_k=payload.get("stage3_best_k"),
        stage4_best_alpha=payload.get("stage4_best_alpha"),
        stage5_best_reranker_input_count=payload.get("stage5_best_reranker_input_count"),
    )


def _provider_for_model(model: str) -> Tuple[str, str]:
    for provider in EMBEDDING_PROVIDERS:
        if provider.model == model:
            return provider.model, provider.backend
    raise KeyError(f"No embedding provider registered for model: {model}")


def _resolve_embedding(row: FormalRunSpec, embedding_model: str) -> FormalRunSpec:
    model, backend = _provider_for_model(embedding_model)
    return replace(
        row,
        embedding_model=model,
        embedding_backend=backend,
        selection_rule="",
    )


def _rank_index(run_id: str) -> int:
    match = re.search(r"top(\d+)", run_id)
    if not match:
        raise ValueError(f"Cannot resolve ranked selection from run_id={run_id}")
    return int(match.group(1)) - 1


def _stage5_multiplier(run_id: str) -> int:
    for multiplier in RERANKER_INPUT_MULTIPLIERS:
        if run_id.endswith(f"{multiplier}k"):
            return multiplier
    raise ValueError(f"Cannot resolve reranker multiplier from run_id={run_id}")


def is_run_resolved(row: FormalRunSpec) -> bool:
    """Return true when a formal row has concrete executable values."""
    if row.selection_rule:
        return False
    if row.embedding_model is None or row.embedding_backend is None or row.k is None:
        return False
    if row.pipeline == "advanced_rag":
        return (
            row.alpha is not None
            and row.reranker_input_count is not None
            and row.reranker_output_count is not None
        )
    return True


def resolve_stage_runs(
    stage: str,
    matrix: Sequence[FormalRunSpec],
    state: FormalSelectionState,
) -> List[FormalRunSpec]:
    """Resolve template rows for a stage using previous-stage selections."""
    rows = [row for row in matrix if row.stage == stage]
    if not rows:
        raise KeyError(f"Unknown formal stage: {stage}")
    if all(is_run_resolved(row) for row in rows):
        return rows

    resolved: List[FormalRunSpec] = []
    for row in rows:
        if stage == "2_k_screening":
            rank = _rank_index(row.run_id)
            if rank >= len(state.stage1_top_embeddings):
                raise RuntimeError("Stage 2 requires completed stage 1 embedding selections")
            resolved.append(_resolve_embedding(row, state.stage1_top_embeddings[rank]))
            continue

        if stage == "3_advanced_review":
            rank = _rank_index(row.run_id)
            if rank >= len(state.stage2_top_pairs):
                raise RuntimeError("Stage 3 requires completed stage 2 embedding/k selections")
            embedding_model, k = state.stage2_top_pairs[rank]
            resolved_row = _resolve_embedding(row, embedding_model)
            resolved.append(
                replace(
                    resolved_row,
                    k=k,
                    reranker_input_count=BASELINE_RERANKER_MULTIPLIER * k,
                    reranker_output_count=k,
                )
            )
            continue

        if stage == "4_alpha_ablation":
            if state.stage3_best_embedding is None or state.stage3_best_k is None:
                raise RuntimeError("Stage 4 requires completed stage 3 selection")
            resolved_row = _resolve_embedding(row, state.stage3_best_embedding)
            resolved.append(
                replace(
                    resolved_row,
                    k=state.stage3_best_k,
                    reranker_input_count=BASELINE_RERANKER_MULTIPLIER
                    * state.stage3_best_k,
                    reranker_output_count=state.stage3_best_k,
                )
            )
            continue

        if stage == "5_reranker_input_ablation":
            if (
                state.stage3_best_embedding is None
                or state.stage3_best_k is None
                or state.stage4_best_alpha is None
            ):
                raise RuntimeError("Stage 5 requires completed stage 4 selection")
            multiplier = _stage5_multiplier(row.run_id)
            resolved_row = _resolve_embedding(row, state.stage3_best_embedding)
            resolved.append(
                replace(
                    resolved_row,
                    k=state.stage3_best_k,
                    alpha=state.stage4_best_alpha,
                    reranker_input_count=multiplier * state.stage3_best_k,
                    reranker_output_count=state.stage3_best_k,
                )
            )
            continue

        if not is_run_resolved(row):
            raise RuntimeError(f"Formal row remains unresolved: {row.run_id}")
        resolved.append(row)

    return resolved


def _query_cache_id(row: FormalRunSpec) -> str:
    if row.embedding_backend not in LOCAL_EMBEDDING_BACKENDS:
        return row.run_id
    if row.pipeline == "advanced_rag":
        if row.embedding_backend == "local_medcpt":
            return "advanced_medcpt_rewritten_query"
        return f"{row.run_id}__{_slug(row.embedding_model)}"
    if row.embedding_backend == "local_medcpt":
        return "stage1_naive_medcpt"
    return f"{row.run_id}__{_slug(row.embedding_model)}"


def _formal_metadata(row: FormalRunSpec, index_path: Path) -> Dict[str, Any]:
    return {
        **asdict(row),
        "index_path": str(index_path),
        "query_cache_id": _query_cache_id(row),
    }


def question_file_for_split(dataset_split: str) -> Path:
    """Resolve the benchmark file once so all formal entrypoints share split semantics."""
    if dataset_split == "dev":
        return MEDQA_USMLE_DEV_FILE
    if dataset_split == "test":
        return MEDQA_USMLE_TEST_FILE
    raise ValueError(f"Unsupported formal dataset_split: {dataset_split}")


def _formal_metadata_for_config(
    row: FormalRunSpec,
    index_path: Path,
    config: FormalExecutionConfig,
) -> Dict[str, Any]:
    metadata = _formal_metadata(row, index_path)
    metadata["dataset_split"] = config.dataset_split
    return metadata


def build_naive_eval_config(
    row: FormalRunSpec,
    index_path: Path,
    config: FormalExecutionConfig,
) -> NaiveRAGEvalConfig:
    return NaiveRAGEvalConfig(
        dev_size=0,
        test_size=config.max_questions,
        manual_top_k=row.k,
        vector_store_path=index_path,
        question_file=question_file_for_split(config.dataset_split),
        concurrency=ConcurrencyConfig(max_concurrent=FORMAL_GENERATOR_MAX_CONCURRENT),
        formal_run_id=row.run_id,
        formal_metadata=_formal_metadata_for_config(row, index_path, config),
    )


def build_enhanced_eval_config(
    row: FormalRunSpec,
    index_path: Path,
    config: FormalExecutionConfig,
) -> EnhancedEvaluationConfig:
    return EnhancedEvaluationConfig(
        dev_size=0,
        test_size=config.max_questions,
        top_k=int(row.k),
        retrieval_top_k=int(row.reranker_input_count),
        reranker_top_k=int(row.reranker_output_count),
        hybrid_alpha=float(row.alpha),
        vector_store_path=index_path,
        question_file=question_file_for_split(config.dataset_split),
        use_query_rewrite=row.query_enhancement_setting == "on",
        concurrency=ConcurrencyConfig(max_concurrent=FORMAL_GENERATOR_MAX_CONCURRENT),
        formal_run_id=row.run_id,
        formal_metadata=_formal_metadata_for_config(row, index_path, config),
    )


def _extract_run_metrics(row: FormalRunSpec, result: Dict[str, Any]) -> Dict[str, Any]:
    test_results = result.get("test_results", result)
    return {
        "run_id": row.run_id,
        "status": str(test_results.get("status", "completed")),
        "stage": row.stage,
        "pipeline": row.pipeline,
        "dataset_split": row.dataset_split,
        "corpus_version": row.corpus_version,
        "embedding_model": row.embedding_model,
        "embedding_backend": row.embedding_backend,
        "k": row.k,
        "alpha": row.alpha,
        "reranker_input_count": row.reranker_input_count,
        "reranker_output_count": row.reranker_output_count,
        "accuracy": float(test_results.get("accuracy", 0.0)),
        "correct": int(test_results.get("correct", 0)),
        "processed_questions": int(test_results.get("processed_questions", 0)),
        "total_questions": int(test_results.get("total_questions", 0)),
        "failed_generator_questions": int(
            test_results.get("failed_generator_questions", 0)
        ),
        "generator_error_question_ids": list(
            test_results.get("generator_error_question_ids", [])
        ),
        "elapsed_time": float(test_results.get("elapsed_time", 0.0)),
    }


async def execute_formal_run(
    row: FormalRunSpec,
    config: FormalExecutionConfig,
) -> Dict[str, Any]:
    """Skip completed runs or dispatch one resolved row to the existing evaluator."""
    if not is_run_resolved(row):
        raise RuntimeError(f"Cannot execute unresolved formal row: {row.run_id}")

    metrics_path = run_dir(row.run_id) / "metrics.json"
    manifest_path = run_dir(row.run_id) / "manifest.json"
    if config.resume and not config.force and metrics_path.exists() and manifest_path.exists():
        manifest = load_json_safe(manifest_path)
        if manifest.get("status") == "completed":
            return load_json_safe(metrics_path)

    index_path = ensure_formal_index(row)
    if row.pipeline == "naive_rag":
        result = await run_complete_evaluation(build_naive_eval_config(row, index_path, config))
    elif row.pipeline == "advanced_rag":
        from app.rag.evaluation.enhanced_rag_eval import run_enhanced_evaluation

        result = await run_enhanced_evaluation(
            build_enhanced_eval_config(row, index_path, config)
        )
    else:
        raise ValueError(f"Unsupported formal pipeline: {row.pipeline}")

    metrics = _extract_run_metrics(row, result)
    if not metrics_path.exists():
        write_metrics(row.run_id, metrics)
    if not manifest_path.exists():
        write_run_manifest(
            row.run_id,
            {**asdict(row), "status": "completed", "metrics_path": str(metrics_path)},
        )
    return metrics


def _is_run_complete(metric: Dict[str, Any]) -> bool:
    """Return ``True`` when a formal run processed every question with at
    most a negligible number of transient generator failures.

    The evaluation pipeline excludes generator-error questions from accuracy
    rather than counting them wrong, so a handful of transient failures (rate
    limits, content filters) should not block the entire stage.  Older metrics
    that lack a ``status`` field are judged solely by the processed/error
    counts.
    """
    total = metric.get("total_questions", 0)
    processed = metric.get("processed_questions", 0)
    failed = metric.get("failed_generator_questions", 0)
    if processed != total:
        return False
    # Tolerate up to 5 generator errors per run (≈0.4% of 1272 dev set).
    return failed <= max(5, total // 250)


def _rank_runs(run_metrics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(run_metrics, key=lambda metric: metric.get("accuracy", 0.0), reverse=True)


def _update_selection_state(
    stage: str,
    rows: Sequence[FormalRunSpec],
    run_metrics: Sequence[Dict[str, Any]],
    state: FormalSelectionState,
) -> None:
    ranked = _rank_runs(run_metrics)
    rows_by_id = {row.run_id: row for row in rows}
    if stage == "1_embedding_screening":
        state.stage1_top_embeddings = [
            str(rows_by_id[metric["run_id"]].embedding_model) for metric in ranked[:2]
        ]
    elif stage == "2_k_screening":
        state.stage2_top_pairs = [
            (
                str(rows_by_id[metric["run_id"]].embedding_model),
                int(rows_by_id[metric["run_id"]].k),
            )
            for metric in ranked[:2]
        ]
    elif stage == "3_advanced_review" and ranked:
        best = rows_by_id[ranked[0]["run_id"]]
        state.stage3_best_embedding = best.embedding_model
        state.stage3_best_k = best.k
    elif stage == "4_alpha_ablation" and ranked:
        state.stage4_best_alpha = rows_by_id[ranked[0]["run_id"]].alpha
    elif stage == "5_reranker_input_ablation" and ranked:
        state.stage5_best_reranker_input_count = rows_by_id[
            ranked[0]["run_id"]
        ].reranker_input_count


def write_stage_checkpoint(
    stage: str,
    rows: Sequence[FormalRunSpec],
    run_metrics: Sequence[Dict[str, Any]],
    state: FormalSelectionState,
) -> None:
    save_json_atomic(
        _stage_checkpoint_path(stage),
        {
            "stage": stage,
            "status": "running",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rows": [asdict(row) for row in rows],
            "run_metrics": list(run_metrics),
            "selection_state": _state_payload(state),
        },
        indent=2,
        ensure_ascii=False,
    )


def write_stage_summary(
    stage: str,
    rows: Sequence[FormalRunSpec],
    run_metrics: Sequence[Dict[str, Any]],
    state: FormalSelectionState,
) -> Dict[str, Any]:
    summary = {
        "stage": stage,
        "status": "completed",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rows": [asdict(row) for row in rows],
        "run_metrics": list(run_metrics),
        "selection_state": _state_payload(state),
    }
    save_json_atomic(_stage_summary_path(stage), summary, indent=2, ensure_ascii=False)
    return summary


async def run_formal_ablation(
    config: FormalExecutionConfig = FormalExecutionConfig(),
) -> Dict[str, Any]:
    """Execute formal ablation stages and write stage-level checkpoints."""
    ensure_data_directories()
    matrix = build_formal_matrix()
    root = _stage_root()
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "formal_ablation_matrix.csv", matrix)
    save_json_atomic(
        root / "formal_ablation_cache_manifest.json",
        build_cache_manifest(matrix),
        indent=2,
        ensure_ascii=False,
    )
    execution_manifest = {
        "run_id": RUN_ID,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "stages": [],
    }
    save_json_atomic(root / "execution_manifest.json", execution_manifest)

    state = FormalSelectionState()
    stage_summaries: List[Dict[str, Any]] = []
    for stage in config.run_stages:
        summary_path = _stage_summary_path(stage)
        if config.resume and not config.force and summary_path.exists():
            summary = load_json_safe(summary_path)
            if summary.get("status") == "completed":
                state = _state_from_payload(summary.get("selection_state", {}))
                stage_summaries.append(summary)
                print(f"[formal] skip completed stage: {stage}", flush=True)
                continue

        rows = resolve_stage_runs(stage, matrix, state)
        run_metrics: List[Dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            print(
                f"[formal][{stage}] run {index}/{len(rows)}: {row.run_id}",
                flush=True,
            )
            run_metrics.append(await execute_formal_run(row, config))
            write_stage_checkpoint(stage, rows, run_metrics, state)

        incomplete_runs = [
            metric
            for metric in run_metrics
            if not _is_run_complete(metric)
        ]
        if incomplete_runs:
            failed_ids = ", ".join(str(metric["run_id"]) for metric in incomplete_runs)
            raise RuntimeError(
                f"Formal stage {stage} has incomplete generator outputs: {failed_ids}. "
                "Check generator_errors.jsonl and rerun before stage selection."
            )

        _update_selection_state(stage, rows, run_metrics, state)
        stage_summaries.append(write_stage_summary(stage, rows, run_metrics, state))

    execution_manifest.update(
        {
            "status": "completed",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "selection_state": _state_payload(state),
            "stages": stage_summaries,
        }
    )
    save_json_atomic(root / "execution_manifest.json", execution_manifest)
    return execution_manifest


def main() -> None:
    manifest = asyncio.run(run_formal_ablation())
    print("=" * 60)
    print("Formal Ablation Execution Complete")
    print("=" * 60)
    print(f"Status: {manifest['status']}")
    print(f"Stages: {len(manifest['stages'])}")
    print(f"Execution manifest: {_stage_root() / 'execution_manifest.json'}")


if __name__ == "__main__":
    main()
