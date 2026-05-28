"""Adapter for the local MedQA-USMLE jsonl splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from ..data_paths import MEDQA_USMLE_DEV_FILE, MEDQA_USMLE_TEST_FILE


OPTION_LABELS = ("A", "B", "C", "D", "E")


def _normalize_options(options: Any) -> List[str]:
    """Keep one option normalization path for dict and list dataset variants."""
    if isinstance(options, Mapping):
        return [str(options[label]) for label in OPTION_LABELS if label in options]
    if isinstance(options, list):
        return [str(option) for option in options]
    raise ValueError(f"Unsupported MedQA-USMLE options payload: {type(options).__name__}")


def _answer_index(answer_idx: Any, options: List[str]) -> int:
    if isinstance(answer_idx, str) and answer_idx.strip():
        label = answer_idx.strip().upper()
        if label in OPTION_LABELS:
            return OPTION_LABELS.index(label)
    if isinstance(answer_idx, int):
        return answer_idx
    raise ValueError(f"Unsupported MedQA-USMLE answer_idx: {answer_idx!r}")


def normalize_medqa_usmle_item(
    item: Mapping[str, Any],
    *,
    split: str,
    row_id: int,
) -> Dict[str, Any]:
    """Return the shared evaluation shape while preserving source metadata."""
    options = _normalize_options(item.get("options"))
    answer_index = _answer_index(item.get("answer_idx"), options)
    if answer_index < 0 or answer_index >= len(options):
        raise ValueError(f"answer_idx is outside options for {split} row {row_id}")

    return {
        "id": f"{split}-{row_id}",
        "question": str(item["question"]),
        "options": options,
        "answer": str(item.get("answer", "")),
        "answer_index": answer_index,
        "answer_idx": OPTION_LABELS[answer_index],
        "meta_info": item.get("meta_info"),
        "split": split,
    }


def load_medqa_usmle_jsonl(path: Path, *, split: str) -> List[Dict[str, Any]]:
    """Load one MedQA-USMLE jsonl split without touching legacy medqa.json."""
    if not path.exists():
        raise FileNotFoundError(f"MedQA-USMLE split not found: {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_id, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(f"MedQA-USMLE {split} row {row_id} is not an object")
            records.append(
                normalize_medqa_usmle_item(payload, split=split, row_id=row_id)
            )
    return records


def load_medqa_usmle_split(split: str) -> List[Dict[str, Any]]:
    """Load the configured local dev/test split by name."""
    split_name = split.strip().lower()
    paths = {
        "dev": MEDQA_USMLE_DEV_FILE,
        "test": MEDQA_USMLE_TEST_FILE,
    }
    if split_name not in paths:
        raise KeyError(f"Unknown MedQA-USMLE split {split!r}; expected dev or test")
    return load_medqa_usmle_jsonl(paths[split_name], split=split_name)


def load_medqa_usmle_counts() -> Dict[str, int]:
    """Summarize split sizes for run manifests without loading legacy data."""
    return {
        "dev": len(load_medqa_usmle_split("dev")),
        "test": len(load_medqa_usmle_split("test")),
    }
