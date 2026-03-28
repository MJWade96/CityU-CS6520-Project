"""
Combine registered corpora into a single dataset for indexing.

The script reuses the registry in ``app.rag.corpus_registry`` so source file
definitions and normalization logic live in one place instead of being
duplicated across scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from app.rag.corpus_registry import CORPUS_REGISTRY, combine_registered_corpora
from app.rag.data_paths import COMBINED_CORPUS_FILE, ensure_data_directories


OUTPUT_FILE = COMBINED_CORPUS_FILE
SELECTED_SOURCES = list(CORPUS_REGISTRY.keys())
TEXTBOOKS_FILE = None
PUBMED_FILE = None
STATPEARLS_FILE = None


def save_combined_corpus(
    output_file: Path,
    source_files: Optional[Dict[str, str]] = None,
    selected_sources: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Combine configured corpora and persist the merged JSON file."""
    ensure_data_directories()
    result = combine_registered_corpora(
        source_files=source_files,
        selected_sources=selected_sources,
    )
    records = result["records"]

    if not records:
        raise FileNotFoundError(
            "No corpus records were loaded from the selected sources."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    result["output_file"] = str(output_file)
    return result


def print_summary(result: Dict[str, object]) -> None:
    """Print a compact summary of the merged corpus."""
    stats = result["stats"]
    records = result["records"]

    print("=" * 60)
    print("Combined Medical Corpora")
    print("=" * 60)
    for source_name, source_stats in stats.items():
        if source_stats["loaded"]:
            print(
                f"{source_name:12} {source_stats['count']:>8,}  {source_stats['path']}"
            )
        else:
            print(f"{source_name:12} {'missing':>8}  not found")

    print("-" * 60)
    print(f"Total chunks: {len(records):,}")
    print(f"Saved to: {result['output_file']}")


def main() -> None:
    source_overrides = {
        "textbooks": TEXTBOOKS_FILE,
        "pubmed": PUBMED_FILE,
        "statpearls": STATPEARLS_FILE,
    }
    result = save_combined_corpus(
        output_file=Path(OUTPUT_FILE),
        source_files={k: v for k, v in source_overrides.items() if v},
        selected_sources=SELECTED_SOURCES,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
