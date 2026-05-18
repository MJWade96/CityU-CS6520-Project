"""Shared evaluation configuration objects used by the native RAG flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..data.data_paths import (
    EVALUATION_RESULTS_DIR,
    FAISS_INDEX_DIR,
    MEDQA_FILE,
)
from .eval_shared import ConcurrencyConfig, EvaluationLLMConfig


SAMPLE_SIZE = 50
TOP_K = 3
DEV_SIZE = 0


@dataclass
class NaiveRAGEvalConfig:
    dev_size: int = 300
    test_size: Optional[int] = None
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    manual_top_k: Optional[int] = 3
    vector_store_path: Path = FAISS_INDEX_DIR
    question_file: Path = MEDQA_FILE
    output_dir: Path = EVALUATION_RESULTS_DIR
    llm: EvaluationLLMConfig = field(default_factory=EvaluationLLMConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


@dataclass(frozen=True)
class EvaluationRunNames:
    artifact_prefix: str = "naive_rag_eval"
    run_name: str = "NAIVE_RAG"
    evaluation_type: str = "NAIVE_RAG"
    dev_script_name: str = "complete_eval_dev"
    test_script_name: str = "complete_eval_test"


@dataclass
class SampleEvalConfig:
    sample_size: int = SAMPLE_SIZE
    top_k: int = TOP_K
    dev_size: int = DEV_SIZE
    question_file: Path = MEDQA_FILE
    output_dir: Path = EVALUATION_RESULTS_DIR
    vector_store_path: Path = FAISS_INDEX_DIR
    llm: EvaluationLLMConfig = field(
        default_factory=lambda: EvaluationLLMConfig(enable_thinking=None)
    )
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
