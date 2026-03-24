"""
Download and prepare the StatPearls corpus.

The heavy lifting lives in ``app.rag.statpearls_dataset`` so the same
download/chunking logic can be reused by other scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.data_paths import CORPUS_DIR, ensure_data_directories
from app.rag.statpearls_dataset import build_statpearls_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and process StatPearls")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=CORPUS_DIR,
        help="Corpus directory that will contain the statpearls folder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data_directories()
    result = build_statpearls_dataset(Path(args.data_dir))

    print("=" * 60)
    print("StatPearls Download Complete")
    print("=" * 60)
    print(f"Archive: {result['archive_path']}")
    print(f"Extracted articles: {result['article_count']:,}")
    print(f"Generated chunks: {result['chunk_count']:,}")
    print(f"Combined corpus: {result['combined_file']}")


if __name__ == "__main__":
    main()
