"""
Centralized project-relative data paths.

The helpers here avoid repeating path-building logic across download,
indexing, and evaluation scripts.
"""

from __future__ import annotations

import os
from pathlib import Path


def _detect_project_root() -> Path:
    """Find the repository root for this package."""
    p = Path(__file__).resolve()
    for candidate in p.parents:
        if (candidate / "app").exists() and (candidate / "data").exists():
            return candidate
    return p.parents[3] if len(p.parents) > 3 else p.parents[-1]


PROJECT_ROOT = _detect_project_root()
PACKAGE_DATA_DIR = PROJECT_ROOT / "app" / "rag" / "data"
DEFAULT_DATA_DIR = PACKAGE_DATA_DIR
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()
RESULTS_DIR = Path(
    os.environ.get("RAG_RESULTS_DIR", str(PROJECT_ROOT / "results"))
).resolve()

CORPUS_DIR = DATA_DIR / "medical_corpus"
EVALUATION_DIR = Path(
    os.environ.get(
        "RAG_EVALUATION_DIR",
        str(PROJECT_ROOT / "app" / "rag" / "evaluation"),
    )
).resolve()
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

COMBINED_CORPUS_FILE = CORPUS_DIR / "combined_corpus.json"
MEDQA_FILE = EVALUATION_DIR / "medqa.json"
FAISS_INDEX_DIR = VECTOR_STORE_DIR / "faiss_index"
EVALUATION_RESULTS_DIR = RESULTS_DIR / "evaluation"
RETRIEVAL_CACHE_DIR = EVALUATION_RESULTS_DIR / "retrieval_cache"


def ensure_data_directories() -> None:
    """Create the standard data directories when they do not exist."""
    for directory in (
        DATA_DIR,
        RESULTS_DIR,
        CORPUS_DIR,
        EVALUATION_DIR,
        VECTOR_STORE_DIR,
        EVALUATION_RESULTS_DIR,
        RETRIEVAL_CACHE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
