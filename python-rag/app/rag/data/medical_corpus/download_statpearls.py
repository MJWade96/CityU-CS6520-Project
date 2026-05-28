"""Thin entrypoint for the existing StatPearls processor."""

from __future__ import annotations

from pathlib import Path

try:
    from .statpearls import process_statpearls
except ImportError:
    from statpearls import process_statpearls


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "statpearls_NBK430685"
OUTPUT_DIR = SCRIPT_DIR / "chunk"


def main() -> None:
    process_statpearls(str(SOURCE_DIR), output_dir=str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
