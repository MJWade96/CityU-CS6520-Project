"""Registered MedRAG-compatible corpus loading for phase 1 experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .data_paths import COMBINED_CORPUS_FILE, TEXTBOOKS_CORPUS_FILE
from .json_utils import load_json_safe


@dataclass(frozen=True)
class CorpusSpec:
    """Single source definition so scripts do not duplicate corpus conventions."""

    name: str
    default_path: Path
    source_label: str


CORPUS_REGISTRY: Dict[str, CorpusSpec] = {
    "statpearls": CorpusSpec(
        name="statpearls",
        default_path=COMBINED_CORPUS_FILE,
        source_label="statpearls",
    ),
    "textbooks": CorpusSpec(
        name="textbooks",
        default_path=TEXTBOOKS_CORPUS_FILE,
        source_label="textbooks",
    ),
}


def _load_json_or_jsonl(path: Path) -> List[Mapping[str, Any]]:
    """Read list-style JSON or JSONL corpora through one shared parser."""
    if not path.exists():
        raise FileNotFoundError(f"Corpus source not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records: List[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"JSONL row {line_number} is not an object: {path}")
                records.append(payload)
        return records

    payload = load_json_safe(path)
    if not isinstance(payload, list):
        raise ValueError(f"Corpus payload must be a list: {path}")
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"Corpus item {index} is not an object: {path}")
    return payload


def normalize_medrag_record(
    record: Mapping[str, Any],
    *,
    source_name: str,
) -> Dict[str, Any]:
    """Normalize MedRAG snippets once while preserving retrieval text fields."""
    record_id = str(record.get("id") or "").strip()
    title = str(record.get("title") or "").strip()
    content = str(record.get("content") or record.get("text") or "").strip()

    if not record_id:
        raise ValueError(f"Corpus record from {source_name} is missing id")
    if not content:
        raise ValueError(f"Corpus record {record_id} from {source_name} has no content")

    contents = str(record.get("contents") or "").strip()
    if not contents:
        contents = f"{title}. {content}".strip() if title else content

    normalized = {
        "id": record_id,
        "title": title,
        "content": content,
        "contents": contents,
        "source": source_name,
    }
    for optional_field in ("textbook", "article", "section"):
        if optional_field in record:
            normalized[optional_field] = record.get(optional_field)
    return normalized


normalize_corpus_record = normalize_medrag_record


def resolve_corpus_source(
    source_name: str,
    source_files: Optional[Mapping[str, str]] = None,
) -> Path:
    """Resolve the configured path for one registered corpus source."""
    if source_name not in CORPUS_REGISTRY:
        known_sources = ", ".join(sorted(CORPUS_REGISTRY))
        raise KeyError(f"Unknown corpus source {source_name!r}; expected one of {known_sources}")

    override = (source_files or {}).get(source_name)
    return Path(override).resolve() if override else CORPUS_REGISTRY[source_name].default_path


def load_registered_corpus(
    source_name: str,
    source_files: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Load and normalize one registered corpus source."""
    spec = CORPUS_REGISTRY[source_name]
    source_path = resolve_corpus_source(source_name, source_files)
    raw_records = _load_json_or_jsonl(source_path)
    return [
        normalize_medrag_record(record, source_name=spec.source_label)
        for record in raw_records
    ]


def combine_registered_corpora(
    source_files: Optional[Mapping[str, str]] = None,
    selected_sources: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Combine selected registered sources and return records plus source stats."""
    sources = list(selected_sources or CORPUS_REGISTRY.keys())
    records: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, object]] = {}

    for source_name in sources:
        source_path = resolve_corpus_source(source_name, source_files)
        source_records = load_registered_corpus(source_name, source_files)
        records.extend(source_records)
        stats[source_name] = {
            "loaded": True,
            "count": len(source_records),
            "path": str(source_path),
        }

    return {
        "records": records,
        "stats": stats,
        "selected_sources": sources,
    }
