"""CLI entrypoint for syncing the MedRAG Textbooks corpus."""

from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    from pathlib import Path as _Path
    import sys as _sys

    _project_root = next(
        parent for parent in _Path(__file__).resolve().parents if (parent / "app").exists()
    )
    _sys.path.insert(0, str(_project_root))

from app.rag.data.data_paths import ensure_data_directories
from app.rag.data.textbooks_dataset import sync_textbooks_dataset


def main() -> None:
    ensure_data_directories()
    result = sync_textbooks_dataset()

    print("=" * 60)
    print("Textbooks Download Complete")
    print("=" * 60)
    print(f"Dataset: {result['dataset']}")
    print(f"Records: {result['record_count']:,}")
    print(f"Output: {result['output_file']}")
    print(f"Cache: {result['cache_dir']}")


if __name__ == "__main__":
    main()
