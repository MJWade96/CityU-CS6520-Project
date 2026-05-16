"""
Centralized project-relative data paths.

The helpers here avoid repeating path-building logic across download,
indexing, and evaluation scripts.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    PROJECT_ROOT / "data"
    if (PROJECT_ROOT / "data").exists()
    else PROJECT_ROOT.parent / "RAG_Medical_Data"
)
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()
RESULTS_DIR = Path(os.environ.get("RAG_RESULTS_DIR", str(PROJECT_ROOT / "results"))).resolve()

CORPUS_DIR = DATA_DIR / "corpus"
EVALUATION_DIR = DATA_DIR / "evaluation"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Centralize frequently reused evaluation artifacts so scripts do not rebuild paths inline.
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
        CORPUS_DIR / "textbooks",
        CORPUS_DIR / "pubmed",
        CORPUS_DIR / "statpearls",
    ):
        directory.mkdir(parents=True, exist_ok=True)
