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
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
EVALUATION_DIR = Path(
    os.environ.get(
        "RAG_EVALUATION_DIR",
        str(PROJECT_ROOT / "app" / "rag" / "evaluation"),
    )
).resolve()
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

TEXTBOOKS_CORPUS_FILE = CORPUS_DIR / "textbooks_corpus.json"
TEXTBOOKS_DOWNLOAD_DIR = CORPUS_DIR / "medrag_textbooks"
COMBINED_CORPUS_FILE = CORPUS_DIR / "combined_corpus.json"
MEDQA_FILE = EVALUATION_DIR / "medqa.json"
MEDQA_USMLE_DIR = BENCHMARKS_DIR / "MedQA-USMLE"
MEDQA_USMLE_DEV_FILE = MEDQA_USMLE_DIR / "dev.jsonl"
MEDQA_USMLE_TEST_FILE = MEDQA_USMLE_DIR / "test.jsonl"
FAISS_INDEX_DIR = VECTOR_STORE_DIR / "faiss_index"
PHASE1_INDEX_DIR = VECTOR_STORE_DIR / "phase1"
EVALUATION_RESULTS_DIR = RESULTS_DIR / "evaluation"
PHASE1_RESULTS_DIR = EVALUATION_RESULTS_DIR / "phase1"
RESULT_INDEXES_DIR = RESULTS_DIR / "indexes"
RETRIEVAL_CACHE_DIR = RESULTS_DIR / "retrieval_cache"
RERANK_CACHE_DIR = RESULTS_DIR / "rerank_cache"
RUNS_DIR = RESULTS_DIR / "runs"


def ensure_data_directories() -> None:
    """Create the standard data directories when they do not exist."""
    for directory in (
        DATA_DIR,
        RESULTS_DIR,
        CORPUS_DIR,
        BENCHMARKS_DIR,
        TEXTBOOKS_DOWNLOAD_DIR,
        EVALUATION_DIR,
        VECTOR_STORE_DIR,
        PHASE1_INDEX_DIR,
        EVALUATION_RESULTS_DIR,
        PHASE1_RESULTS_DIR,
        RESULT_INDEXES_DIR,
        RETRIEVAL_CACHE_DIR,
        RERANK_CACHE_DIR,
        RUNS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
