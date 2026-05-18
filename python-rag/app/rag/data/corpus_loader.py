"""Shared combined-corpus loading helpers for both index builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from .json_utils import load_json_safe


CORPUS_METADATA_FIELDS = (
    "id",
    "title",
    "source",
    "textbook",
    "pmid",
    "journal",
    "year",
)


def load_corpus_chunks(corpus_file: Path) -> List[Dict[str, Any]]:
    """Load raw corpus chunks once so each stack can build its own document type."""
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    chunks = load_json_safe(corpus_file)
    if not isinstance(chunks, list):
        raise ValueError(f"Corpus payload must be a list: {corpus_file}")
    return chunks


def build_corpus_metadata(chunk: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep shared metadata mapping in one place to avoid duplicate index builders."""
    return {field: chunk.get(field, "") for field in CORPUS_METADATA_FIELDS}