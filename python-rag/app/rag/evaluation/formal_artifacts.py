"""Small file helpers for formal ablation artifacts.

The formal evaluators share these write/read helpers so JSONL and manifest
handling is not duplicated across Naive and Advanced branches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set

from app.rag.data.data_paths import RERANK_CACHE_DIR, RETRIEVAL_CACHE_DIR, RUNS_DIR
from app.rag.data.json_utils import save_json_atomic


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON row while keeping directory creation in one place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def checkpoint_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.checkpoint{path.suffix}")


def append_jsonl_with_checkpoint(path: Path, row: Mapping[str, Any]) -> None:
    """Append matching final/checkpoint rows without duplicating write logic."""
    append_jsonl(path, row)
    append_jsonl(checkpoint_path(path), row)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file, returning an empty list when no artifact exists yet."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def completed_question_ids(path: Path) -> Set[str]:
    """Return question ids already present in a checkpoint or final JSONL file."""
    return {
        str(row["question_id"])
        for row in load_jsonl(path)
        if row.get("question_id") is not None
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist JSON through the project's atomic writer."""
    save_json_atomic(path, dict(payload), indent=2, ensure_ascii=False)


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def retrieval_cache_dir(cache_id: str) -> Path:
    return RETRIEVAL_CACHE_DIR / cache_id


def rerank_cache_dir(cache_id: str) -> Path:
    return RERANK_CACHE_DIR / cache_id


def write_run_manifest(run_id: str, payload: Mapping[str, Any]) -> None:
    write_json(run_dir(run_id) / "manifest.json", payload)


def write_metrics(run_id: str, metrics: Mapping[str, Any]) -> None:
    write_json(run_dir(run_id) / "metrics.json", metrics)
