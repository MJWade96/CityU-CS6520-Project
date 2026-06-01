"""Formal ablation artifact and checkpoint writers.

The recorder centralizes JSON/JSONL persistence so naive and enhanced
evaluators can add formal audit outputs without duplicating file handling.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from ..data.data_paths import RERANK_CACHE_DIR, RESULTS_DIR, RETRIEVAL_CACHE_DIR, RUNS_DIR
from ..data.json_utils import load_json_safe, save_json_atomic


CONFIG_VERSION = "formal_ablation_v1"


@dataclass(frozen=True)
class FormalRunMetadata:
    """Metadata needed to reproduce one formal ablation run."""

    run_id: str
    stage: str
    pipeline: str
    corpus_version: str
    embedding_model: str
    embedding_backend: str
    faiss_index_type: str
    k: int
    alpha: Optional[float]
    reranker_input_count: Optional[int]
    reranker_output_count: Optional[int]
    query_enhancement_setting: str
    generator_model: str
    prompt_version: str
    dataset_split: str
    random_seed: int
    cache_ids: Dict[str, str]


@dataclass
class CacheGroupManifest:
    """One manifest per cache/run group, matching docs/cache_run_record_plan.md."""

    artifact_id: str
    artifact_group: str
    status: str
    files: Dict[str, str]
    input_artifacts: Dict[str, str]
    parameters: Dict[str, Any]
    dataset_split: str
    fingerprint: Dict[str, Any]
    created_at: str
    updated_at: str
    completed_at: Optional[str]
    code_version: Optional[str]
    config_version: str
    expected_row_count: Optional[int]
    row_count: int


def now_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def current_code_version() -> Optional[str]:
    """Best-effort git revision for reproducibility manifests."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _write_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def question_ids_from_jsonl(path: Path) -> set[str]:
    return {
        str(row.get("question_id", ""))
        for row in iter_jsonl(path)
        if str(row.get("question_id", "")).strip()
    }


class FormalArtifactRecorder:
    """Write cache and per-run formal artifacts with resumable checkpoints."""

    RUN_FILES = {
        "selected_contexts": "selected_contexts.jsonl",
        "final_prompts": "final_prompts.jsonl",
        "llm_outputs": "llm_outputs.jsonl",
        "evaluation_outputs": "evaluation_outputs.jsonl",
    }
    RETRIEVAL_FILES = {
        "query_texts": "query_texts.jsonl",
        "retrieval_top10": "retrieval_top10.jsonl",
        "dense_candidates": "dense_candidates.jsonl",
        "sparse_candidates": "sparse_candidates.jsonl",
        "fusion_candidates": "fusion_candidates.jsonl",
        "retrieval_top80": "retrieval_top80.jsonl",
    }
    RERANK_FILES = {"rerank_outputs": "rerank_outputs.jsonl"}

    def __init__(
        self,
        metadata: FormalRunMetadata,
        *,
        results_dir: Path = RESULTS_DIR,
    ) -> None:
        self.metadata = metadata
        self.results_dir = results_dir
        if results_dir == RESULTS_DIR:
            self.run_dir = RUNS_DIR / metadata.run_id
            self.retrieval_dir = RETRIEVAL_CACHE_DIR / metadata.cache_ids["retrieval"]
            self.rerank_dir = RERANK_CACHE_DIR / metadata.cache_ids["rerank"]
        else:
            self.run_dir = results_dir / "runs" / metadata.run_id
            self.retrieval_dir = (
                results_dir / "retrieval_cache" / metadata.cache_ids["retrieval"]
            )
            self.rerank_dir = results_dir / "rerank_cache" / metadata.cache_ids["rerank"]
        self.code_version = current_code_version()

    @property
    def run_manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def run_checkpoint_path(self) -> Path:
        return self.run_dir / "run_checkpoint.json"

    def is_run_completed(self) -> bool:
        if not self.run_manifest_path.exists() or not (self.run_dir / "metrics.json").exists():
            return False
        manifest = load_json_safe(self.run_manifest_path)
        return manifest.get("status") == "completed"

    def start_run(self, *, expected_question_count: int) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.retrieval_dir.mkdir(parents=True, exist_ok=True)
        self.rerank_dir.mkdir(parents=True, exist_ok=True)
        if self.run_manifest_path.exists():
            manifest = load_json_safe(self.run_manifest_path)
            if manifest.get("status") == "completed":
                return
            created_at = str(manifest.get("created_at") or now_timestamp())
        else:
            created_at = now_timestamp()

        manifest = self._manifest(
            artifact_id=self.metadata.run_id,
            artifact_group="run",
            base_dir=self.run_dir,
            files={
                **self.RUN_FILES,
                "metrics": "metrics.json",
                "checkpoint": "run_checkpoint.json",
            },
            expected_row_count=expected_question_count,
            status="running",
            created_at=created_at,
            row_count=len(self.completed_question_ids("evaluation_outputs")),
        )
        save_json_atomic(self.run_manifest_path, asdict(manifest))

    def completed_question_ids(self, artifact_name: str) -> set[str]:
        return question_ids_from_jsonl(self._logical_path(artifact_name))

    def write_query_text(self, row: Mapping[str, Any]) -> None:
        self._append("query_texts", row)

    def write_retrieval(self, artifact_name: str, row: Mapping[str, Any]) -> None:
        if artifact_name not in self.RETRIEVAL_FILES:
            raise KeyError(f"Unknown retrieval artifact: {artifact_name}")
        self._append(artifact_name, row)

    def write_rerank(self, row: Mapping[str, Any]) -> None:
        self._append("rerank_outputs", row)

    def write_selected_contexts(self, row: Mapping[str, Any]) -> None:
        self._append("selected_contexts", row)

    def write_final_prompt(self, row: Mapping[str, Any]) -> None:
        self._append("final_prompts", row)

    def write_llm_output(self, row: Mapping[str, Any]) -> None:
        self._append("llm_outputs", row)

    def write_evaluation_output(self, row: Mapping[str, Any]) -> None:
        self._append("evaluation_outputs", row)

    def save_run_checkpoint(self, payload: Mapping[str, Any]) -> None:
        checkpoint = dict(payload)
        checkpoint["updated_at"] = now_timestamp()
        save_json_atomic(self.run_checkpoint_path, checkpoint)

    def finalize_run(self, metrics: Mapping[str, Any]) -> None:
        self._finalize_group("retrieval", self.retrieval_dir, self.RETRIEVAL_FILES)
        self._finalize_group("rerank", self.rerank_dir, self.RERANK_FILES)
        self._finalize_group("run", self.run_dir, self.RUN_FILES)
        save_json_atomic(self.run_dir / "metrics.json", dict(metrics))
        if self.run_checkpoint_path.exists():
            self.run_checkpoint_path.unlink()

        manifest = self._manifest(
            artifact_id=self.metadata.run_id,
            artifact_group="run",
            base_dir=self.run_dir,
            files={
                **self.RUN_FILES,
                "metrics": "metrics.json",
                "manifest": "manifest.json",
            },
            expected_row_count=metrics.get("total_questions"),
            status="completed",
            created_at=self._created_at(self.run_manifest_path),
            row_count=int(metrics.get("processed_questions", 0)),
            completed_at=now_timestamp(),
        )
        save_json_atomic(self.run_manifest_path, asdict(manifest))

    def _append(self, artifact_name: str, row: Mapping[str, Any]) -> None:
        _write_jsonl(self._checkpoint_path(artifact_name), row)

    def _logical_path(self, artifact_name: str) -> Path:
        if artifact_name in self.RUN_FILES:
            return self.run_dir / self.RUN_FILES[artifact_name]
        if artifact_name in self.RETRIEVAL_FILES:
            return self.retrieval_dir / self.RETRIEVAL_FILES[artifact_name]
        if artifact_name in self.RERANK_FILES:
            return self.rerank_dir / self.RERANK_FILES[artifact_name]
        raise KeyError(f"Unknown artifact: {artifact_name}")

    def _checkpoint_path(self, artifact_name: str) -> Path:
        final_path = self._logical_path(artifact_name)
        return final_path.with_name(final_path.stem + ".checkpoint" + final_path.suffix)

    def _finalize_group(
        self,
        artifact_group: str,
        base_dir: Path,
        files: Mapping[str, str],
    ) -> None:
        for artifact_name in files:
            checkpoint_path = self._checkpoint_path(artifact_name)
            final_path = self._logical_path(artifact_name)
            if checkpoint_path.exists():
                final_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(checkpoint_path), str(final_path))

        manifest_path = base_dir / "manifest.json"
        manifest = self._manifest(
            artifact_id=self._artifact_id_for_group(artifact_group),
            artifact_group=artifact_group,
            base_dir=base_dir,
            files=dict(files),
            expected_row_count=None,
            status="completed",
            created_at=self._created_at(manifest_path),
            row_count=self._group_row_count(files),
            completed_at=now_timestamp(),
        )
        save_json_atomic(manifest_path, asdict(manifest))

    def _artifact_id_for_group(self, artifact_group: str) -> str:
        if artifact_group == "retrieval":
            return self.metadata.cache_ids["retrieval"]
        if artifact_group == "rerank":
            return self.metadata.cache_ids["rerank"]
        return self.metadata.run_id

    def _group_row_count(self, files: Mapping[str, str]) -> int:
        if "evaluation_outputs" in files:
            return len(question_ids_from_jsonl(self.run_dir / files["evaluation_outputs"]))
        first_file = next(iter(files.values()), "")
        if not first_file:
            return 0
        if first_file in self.RETRIEVAL_FILES.values():
            return len(question_ids_from_jsonl(self.retrieval_dir / first_file))
        if first_file in self.RERANK_FILES.values():
            return len(question_ids_from_jsonl(self.rerank_dir / first_file))
        return len(question_ids_from_jsonl(self.run_dir / first_file))

    def _created_at(self, manifest_path: Path) -> str:
        if manifest_path.exists():
            try:
                return str(load_json_safe(manifest_path).get("created_at") or now_timestamp())
            except Exception:
                return now_timestamp()
        return now_timestamp()

    def _manifest(
        self,
        *,
        artifact_id: str,
        artifact_group: str,
        base_dir: Path,
        files: Mapping[str, str],
        expected_row_count: Optional[int],
        status: str,
        created_at: str,
        row_count: int,
        completed_at: Optional[str] = None,
    ) -> CacheGroupManifest:
        return CacheGroupManifest(
            artifact_id=artifact_id,
            artifact_group=artifact_group,
            status=status,
            files={name: str(base_dir / filename) for name, filename in files.items()},
            input_artifacts=dict(self.metadata.cache_ids),
            parameters=asdict(self.metadata),
            dataset_split=self.metadata.dataset_split,
            fingerprint={
                "run_id": self.metadata.run_id,
                "stage": self.metadata.stage,
                "pipeline": self.metadata.pipeline,
            },
            created_at=created_at,
            updated_at=now_timestamp(),
            completed_at=completed_at,
            code_version=self.code_version,
            config_version=CONFIG_VERSION,
            expected_row_count=expected_row_count,
            row_count=row_count,
        )


def cache_id_for_run(run_id: str, suffix: str) -> str:
    """Keep cache-id construction shared between executor and evaluators."""
    return f"{run_id}__{suffix}"


def formal_metadata_from_dict(payload: Mapping[str, Any]) -> FormalRunMetadata:
    """Validate and convert evaluator config metadata into a typed recorder input."""
    return FormalRunMetadata(
        run_id=str(payload["run_id"]),
        stage=str(payload["stage"]),
        pipeline=str(payload["pipeline"]),
        corpus_version=str(payload["corpus_version"]),
        embedding_model=str(payload["embedding_model"]),
        embedding_backend=str(payload["embedding_backend"]),
        faiss_index_type=str(payload["faiss_index_type"]),
        k=int(payload["k"]),
        alpha=None if payload.get("alpha") is None else float(payload["alpha"]),
        reranker_input_count=(
            None
            if payload.get("reranker_input_count") is None
            else int(payload["reranker_input_count"])
        ),
        reranker_output_count=(
            None
            if payload.get("reranker_output_count") is None
            else int(payload["reranker_output_count"])
        ),
        query_enhancement_setting=str(payload["query_enhancement_setting"]),
        generator_model=str(payload["generator_model"]),
        prompt_version=str(payload["prompt_version"]),
        dataset_split=str(payload["dataset_split"]),
        random_seed=int(payload["random_seed"]),
        cache_ids=dict(payload["cache_ids"]),
    )


def make_recorder(
    formal_run_id: Optional[str],
    formal_metadata: Optional[Mapping[str, Any]],
) -> Optional[FormalArtifactRecorder]:
    if formal_run_id is None:
        return None
    if not formal_metadata:
        raise ValueError("formal_metadata is required when formal_run_id is set")
    metadata = formal_metadata_from_dict(formal_metadata)
    if metadata.run_id != formal_run_id:
        raise ValueError(
            f"formal_run_id {formal_run_id!r} does not match metadata run_id {metadata.run_id!r}"
        )
    return FormalArtifactRecorder(metadata)
