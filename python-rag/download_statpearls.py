"""
Download and prepare the StatPearls corpus.

The heavy lifting lives in ``app.rag.statpearls_dataset`` so the same
download/chunking logic can be reused by other scripts.
"""

from __future__ import annotations

from pathlib import Path

from app.rag.data_paths import CORPUS_DIR, ensure_data_directories
from app.rag.statpearls_dataset import build_statpearls_dataset


DATA_DIR = CORPUS_DIR


def main() -> None:
    ensure_data_directories()
    result = build_statpearls_dataset(Path(DATA_DIR))

    print("=" * 60)
    print("StatPearls Download Complete")
    print("=" * 60)
    print(f"Archive: {result['archive_path']}")
    print(f"Extracted articles: {result['article_count']:,}")
    print(f"Generated chunks: {result['chunk_count']:,}")
    print(f"Combined corpus: {result['combined_file']}")


if __name__ == "__main__":
    main()
