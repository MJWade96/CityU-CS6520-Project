"""
Centralized project-relative data paths.

The helpers here avoid repeating path-building logic across download,
indexing, and evaluation scripts.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
LEGACY_DATA_DIR = WORKSPACE_ROOT / "RAG_Medical_Data"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" if (PROJECT_ROOT / "data").exists() else LEGACY_DATA_DIR
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()
RESULTS_DIR = Path(os.environ.get("RAG_RESULTS_DIR", str(DEFAULT_RESULTS_DIR))).resolve()

CORPUS_DIR = DATA_DIR / "corpus"
EVALUATION_DIR = DATA_DIR / "evaluation"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
MEDICAL_KNOWLEDGE_DIR = DATA_DIR / "medical_knowledge"

TEXTBOOKS_DIR = CORPUS_DIR / "textbooks"
PUBMED_DIR = CORPUS_DIR / "pubmed"
STATPEARLS_DIR = CORPUS_DIR / "statpearls"

COMBINED_CORPUS_FILE = CORPUS_DIR / "combined_corpus.json"
MEDQA_FILE = EVALUATION_DIR / "medqa.json"
FAISS_INDEX_DIR = VECTOR_STORE_DIR / "faiss_index"
EVALUATION_RESULTS_DIR = RESULTS_DIR / "evaluation"


def ensure_data_directories() -> None:
    """Create the standard data directories when they do not exist."""
    for directory in (
        DATA_DIR,
        RESULTS_DIR,
        CORPUS_DIR,
        EVALUATION_DIR,
        VECTOR_STORE_DIR,
        EVALUATION_RESULTS_DIR,
        MEDICAL_KNOWLEDGE_DIR,
        TEXTBOOKS_DIR,
        PUBMED_DIR,
        STATPEARLS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def get_corpus_path(*parts: str) -> str:
    return str(CORPUS_DIR.joinpath(*parts))


def get_evaluation_path(*parts: str) -> str:
    return str(EVALUATION_DIR.joinpath(*parts))


def get_vector_store_path(*parts: str) -> str:
    return str(VECTOR_STORE_DIR.joinpath(*parts))


def get_results_path(*parts: str) -> str:
    return str(RESULTS_DIR.joinpath(*parts))


def get_evaluation_results_path(*parts: str) -> str:
    return str(EVALUATION_RESULTS_DIR.joinpath(*parts))
