"""
Corpus registry and normalization helpers.

The registry keeps source definitions in one place so combine/index scripts
do not duplicate file-name or normalization logic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .data_paths import CORPUS_DIR


@dataclass(frozen=True)
class CorpusDefinition:
    name: str
    candidates: List[Path]
    description: str


CORPUS_REGISTRY: Dict[str, CorpusDefinition] = {
    "textbooks": CorpusDefinition(
        name="textbooks",
        candidates=[CORPUS_DIR / "textbooks" / "textbooks_combined.json"],
        description="Chunked medical textbooks.",
    ),
    "pubmed": CorpusDefinition(
        name="pubmed",
        candidates=[
            CORPUS_DIR / "pubmed" / "pubmed_chunks.json",
            CORPUS_DIR / "pubmed" / "pubmed_abstracts.json",
        ],
        description="PubMed abstracts or pre-chunked PubMed corpus.",
    ),
    "statpearls": CorpusDefinition(
        name="statpearls",
        candidates=[CORPUS_DIR / "statpearls" / "statpearls_combined.json"],
        description="Chunked StatPearls articles converted from NCBI Bookshelf NXML.",
    ),
}


def resolve_corpus_file(source_name: str, override: Optional[str] = None) -> Optional[Path]:
    """Resolve a source to an existing file, optionally using an explicit override."""
    if override:
        override_path = Path(override)
        return override_path if override_path.exists() else None

    definition = CORPUS_REGISTRY[source_name]
    for candidate in definition.candidates:
        if candidate.exists():
            return candidate
    return None


def normalize_record(record: Dict, source_name: str, default_id_prefix: str) -> Dict:
    """
    Normalize heterogeneous corpus entries into the shape expected by indexing.

    The normalization is data-driven so combine and indexing code can reuse it
    across multiple corpora instead of repeating source-specific branches.
    """
    title = record.get("title", "") or record.get("file", "") or source_name
    content = record.get("content")
    if not content:
        abstract = record.get("abstract", "")
        content = "\n\n".join(part for part in [title, abstract] if part).strip()
    content = content or ""

    content_digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    normalized = {
        "id": record.get("id") or record.get("pmid") or f"{default_id_prefix}_{content_digest}",
        "title": title,
        "content": content,
        "contents": record.get("contents") or f"{title}. {content}".strip(". "),
        "source": record.get("source") or source_name,
    }

    for field in ("textbook", "pmid", "journal", "year", "authors", "mesh_terms", "file"):
        if field in record and record[field] not in (None, ""):
            normalized[field] = record[field]

    return normalized


def load_corpus_records(source_name: str, path: Path) -> List[Dict]:
    """Load and normalize records from a JSON corpus file."""
    with path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    if isinstance(raw_data, dict) and "records" in raw_data:
        raw_items = raw_data["records"]
    else:
        raw_items = raw_data

    return [
        normalize_record(item, source_name=source_name, default_id_prefix=source_name)
        for item in raw_items
        if isinstance(item, dict)
    ]


def combine_registered_corpora(
    source_files: Optional[Dict[str, str]] = None,
    selected_sources: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """
    Load all available corpora from the registry.

    Returning both the records and per-source metadata keeps callers thin.
    """
    source_files = source_files or {}
    selected = list(selected_sources or CORPUS_REGISTRY.keys())

    combined: List[Dict] = []
    stats: Dict[str, Dict[str, object]] = {}

    for source_name in selected:
        resolved = resolve_corpus_file(source_name, source_files.get(source_name))
        if resolved is None:
            stats[source_name] = {"loaded": False, "count": 0, "path": None}
            continue

        records = load_corpus_records(source_name, resolved)
        combined.extend(records)
        stats[source_name] = {
            "loaded": True,
            "count": len(records),
            "path": str(resolved),
        }

    return {"records": combined, "stats": stats}
