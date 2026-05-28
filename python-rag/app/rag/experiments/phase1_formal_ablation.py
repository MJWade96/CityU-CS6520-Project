"""Formal phase-1 ablation framework.

The runner is intentionally plan-first: it builds the formal experiment matrix
and cache manifest without reusing the old smoke entrypoint or legacy MedQA file.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.rag.data.benchmarks.medqa_usmle import load_medqa_usmle_counts
from app.rag.data.data_paths import (
    MEDQA_FILE,
    MEDQA_USMLE_DEV_FILE,
    MEDQA_USMLE_TEST_FILE,
    RERANK_CACHE_DIR,
    RESULT_INDEXES_DIR,
    RETRIEVAL_CACHE_DIR,
    RUNS_DIR,
    ensure_data_directories,
)
from app.rag.data.json_utils import save_json_atomic
from app.rag.retriever.runtime_config import (
    DEFAULT_EMBEDDING_API_BASE_URL,
    first_env_value,
)


RUN_ID = "phase1_formal_ablation"
EXECUTION_MODE = "plan_only"
RANDOM_SEED = 6520
PROMPT_VERSION = "medical_mcq_v1"
GENERATOR_MODEL = "Qwen3-4B"
FAISS_INDEX_TYPE = "FlatIP"
RETRIEVAL_CACHE_TOP_K = 80
BASELINE_K = 5
BASELINE_ALPHA = 0.5
BASELINE_RERANKER_MULTIPLIER = 4
CORPUS_VARIANTS = {
    "statpearls": ("statpearls",),
    "statpearls_textbooks": ("statpearls", "textbooks"),
}
K_VALUES = (3, 5, 10)
ALPHA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
RERANKER_INPUT_MULTIPLIERS = (2, 4, 8)
CACHE_KEYS = (
    "chunk_embeddings",
    "query_embeddings",
    "faiss_index",
    "retrieval_top80",
    "rerank_outputs",
    "final_prompts",
    "llm_outputs",
    "token_usage",
    "estimated_token_cost",
)


@dataclass(frozen=True)
class EmbeddingProviderSpec:
    """Embedding candidate metadata shared by matrix rows and cache manifests."""

    name: str
    model: str
    backend: str
    deployment: str
    api_base_url: Optional[str] = None
    medscore_reference: Optional[str] = None


EMBEDDING_PROVIDERS = (
    EmbeddingProviderSpec(
        name="bge_m3",
        model="BAAI/bge-m3",
        backend="siliconflow_api",
        deployment="OpenAI-compatible API embedding",
        api_base_url=first_env_value(
            "RAG_EMBEDDING_API_BASE_URL",
            default=DEFAULT_EMBEDDING_API_BASE_URL,
        ),
    ),
    EmbeddingProviderSpec(
        name="medcpt",
        model="ncbi/MedCPT",
        backend="local_medcpt",
        deployment="local offline embedding artifact generation",
        medscore_reference="https://github.com/Heyuan9/MedScore",
    ),
    EmbeddingProviderSpec(
        name="bge_large_en_v1_5",
        model="BAAI/bge-large-en-v1.5",
        backend="siliconflow_api",
        deployment="OpenAI-compatible API embedding",
        api_base_url=first_env_value(
            "RAG_EMBEDDING_API_BASE_URL",
            default=DEFAULT_EMBEDDING_API_BASE_URL,
        ),
    ),
)


@dataclass(frozen=True)
class FormalRunSpec:
    """One formal ablation run row with the required report metadata fields."""

    stage: str
    run_id: str
    pipeline: str
    corpus_version: str
    embedding_model: str
    embedding_backend: str
    faiss_index_type: str
    k: Any
    alpha: Any
    reranker_input_count: Any
    reranker_output_count: Any
    query_enhancement_setting: str
    generator_model: str
    prompt_version: str
    dataset_split: str
    random_seed: int
    selection_rule: str = ""


def _slug(value: Any) -> str:
    return str(value).lower().replace("/", "_").replace(" ", "_").replace(".", "p")


def _provider(name: str) -> EmbeddingProviderSpec:
    for provider in EMBEDDING_PROVIDERS:
        if provider.name == name:
            return provider
    raise KeyError(f"Unknown embedding provider: {name}")


def _run_spec(
    *,
    stage: str,
    run_id: str,
    pipeline: str,
    corpus_version: str = "statpearls_textbooks",
    embedding: str = "bge_m3",
    k: Any = BASELINE_K,
    alpha: Any = None,
    reranker_input_count: Any = None,
    reranker_output_count: Any = None,
    query_enhancement_setting: str = "off",
    selection_rule: str = "",
) -> FormalRunSpec:
    provider = _provider(embedding)
    return FormalRunSpec(
        stage=stage,
        run_id=run_id,
        pipeline=pipeline,
        corpus_version=corpus_version,
        embedding_model=provider.model,
        embedding_backend=provider.backend,
        faiss_index_type=FAISS_INDEX_TYPE,
        k=k,
        alpha=alpha,
        reranker_input_count=reranker_input_count,
        reranker_output_count=reranker_output_count,
        query_enhancement_setting=query_enhancement_setting,
        generator_model=GENERATOR_MODEL,
        prompt_version=PROMPT_VERSION,
        dataset_split="dev",
        random_seed=RANDOM_SEED,
        selection_rule=selection_rule,
    )


def build_formal_matrix() -> List[FormalRunSpec]:
    """Build stages 0-5 without hardcoding duplicate metadata per row."""
    rows: List[FormalRunSpec] = []

    for corpus_name in CORPUS_VARIANTS:
        rows.append(
            _run_spec(
                stage="0_corpus_ablation",
                run_id=f"stage0_naive_{corpus_name}",
                pipeline="naive_rag",
                corpus_version=corpus_name,
                alpha=None,
                reranker_input_count=0,
                reranker_output_count=0,
            )
        )
        rows.append(
            _run_spec(
                stage="0_corpus_ablation",
                run_id=f"stage0_advanced_{corpus_name}",
                pipeline="advanced_rag",
                corpus_version=corpus_name,
                alpha=BASELINE_ALPHA,
                reranker_input_count=f"{BASELINE_RERANKER_MULTIPLIER}k",
                reranker_output_count="k",
                query_enhancement_setting="on",
            )
        )

    for provider in EMBEDDING_PROVIDERS:
        rows.append(
            _run_spec(
                stage="1_embedding_screening",
                run_id=f"stage1_naive_{provider.name}",
                pipeline="naive_rag",
                embedding=provider.name,
                k=BASELINE_K,
                alpha=None,
                reranker_input_count=0,
                reranker_output_count=0,
            )
        )

    for rank in (1, 2):
        for k in K_VALUES:
            rows.append(
                _run_spec(
                    stage="2_k_screening",
                    run_id=f"stage2_naive_stage1_top{rank}_embedding_k{k}",
                    pipeline="naive_rag",
                    embedding="bge_m3",
                    k=k,
                    alpha=None,
                    reranker_input_count=0,
                    reranker_output_count=0,
                    selection_rule=(
                        f"use stage-1 ranked embedding #{rank}; concrete provider "
                        "is resolved after stage 1 results"
                    ),
                )
            )

    for rank in (1, 2):
        rows.append(
            _run_spec(
                stage="3_advanced_review",
                run_id=f"stage3_advanced_stage2_top{rank}_embedding_k",
                pipeline="advanced_rag",
                k=f"stage2_top{rank}_k",
                alpha=BASELINE_ALPHA,
                reranker_input_count=f"{BASELINE_RERANKER_MULTIPLIER}k",
                reranker_output_count="k",
                query_enhancement_setting="on",
                selection_rule=(
                    f"use stage-2 ranked (embedding, k) combination #{rank}; "
                    "concrete provider and k are resolved after stage 2 results"
                ),
            )
        )

    for alpha in ALPHA_VALUES:
        rows.append(
            _run_spec(
                stage="4_alpha_ablation",
                run_id=f"stage4_advanced_alpha_{_slug(alpha)}",
                pipeline="advanced_rag",
                k="best_k",
                alpha=alpha,
                reranker_input_count=f"{BASELINE_RERANKER_MULTIPLIER}k",
                reranker_output_count="k",
                query_enhancement_setting="on",
                selection_rule="use best embedding and k selected from stage 3",
            )
        )

    for multiplier in RERANKER_INPUT_MULTIPLIERS:
        rows.append(
            _run_spec(
                stage="5_reranker_input_ablation",
                run_id=f"stage5_advanced_reranker_input_{multiplier}k",
                pipeline="advanced_rag",
                k="best_k",
                alpha="best_alpha",
                reranker_input_count=f"{multiplier}k",
                reranker_output_count="k",
                query_enhancement_setting="on",
                selection_rule="use best embedding, k, and alpha selected from stage 4",
            )
        )

    return rows


def build_cache_manifest(rows: Sequence[FormalRunSpec]) -> Dict[str, Any]:
    """Declare every reusable artifact recommended for formal ablations."""
    manifest: Dict[str, Any] = {
        "cache_top_k": RETRIEVAL_CACHE_TOP_K,
        "cache_keys": list(CACHE_KEYS),
        "base_dirs": {
            "indexes": str(RESULT_INDEXES_DIR),
            "retrieval_cache": str(RETRIEVAL_CACHE_DIR),
            "rerank_cache": str(RERANK_CACHE_DIR),
            "runs": str(RUNS_DIR),
        },
        "runs": {},
    }
    for row in rows:
        run_dir = RUNS_DIR / row.run_id
        manifest["runs"][row.run_id] = {
            "chunk_embeddings": str(
                RESULT_INDEXES_DIR / row.run_id / "chunk_embeddings.jsonl"
            ),
            "query_embeddings": str(
                RETRIEVAL_CACHE_DIR / row.run_id / "query_embeddings.jsonl"
            ),
            "faiss_index": str(RESULT_INDEXES_DIR / row.run_id / "faiss_index"),
            "retrieval_top80": str(
                RETRIEVAL_CACHE_DIR / row.run_id / "retrieval_top80.jsonl"
            ),
            "rerank_outputs": str(RERANK_CACHE_DIR / row.run_id / "rerank_outputs.jsonl"),
            "final_prompts": str(run_dir / "final_prompts.jsonl"),
            "llm_outputs": str(run_dir / "llm_outputs.jsonl"),
            "token_usage": str(run_dir / "token_usage.json"),
            "estimated_token_cost": str(run_dir / "estimated_token_cost.json"),
        }
    return manifest


def build_final_test_plan() -> Dict[str, Any]:
    """Keep final test comparison separate from dev ablation selection."""
    return {
        "dataset_split": "test",
        "dataset_file": str(MEDQA_USMLE_TEST_FILE),
        "comparisons": [
            {"pipeline": "naive_rag", "status": "requires_final_dev_selection"},
            {"pipeline": "advanced_rag", "status": "requires_final_dev_selection"},
            {
                "pipeline": "graphrag",
                "status": "blocked",
                "reason": "GraphRAG backend is not implemented in the current mainline.",
            },
        ],
    }


def write_csv(path: Path, rows: Iterable[FormalRunSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]
    if not row_dicts:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)


def build_formal_ablation_manifest() -> Dict[str, Any]:
    """Build the formal framework manifest without running accuracy evaluation."""
    rows = build_formal_matrix()
    dataset_counts = load_medqa_usmle_counts()
    return {
        "run_id": RUN_ID,
        "status": "framework_ready_not_executed",
        "execution_mode": EXECUTION_MODE,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "legacy_medqa_file_not_used": str(MEDQA_FILE),
        "dev_split": {
            "role": "ablation_selection",
            "file": str(MEDQA_USMLE_DEV_FILE),
            "question_count": dataset_counts["dev"],
        },
        "test_split": {
            "role": "final_naive_advanced_graphrag_comparison",
            "file": str(MEDQA_USMLE_TEST_FILE),
            "question_count": dataset_counts["test"],
        },
        "embedding_providers": [asdict(provider) for provider in EMBEDDING_PROVIDERS],
        "matrix": [asdict(row) for row in rows],
        "cache_manifest": build_cache_manifest(rows),
        "stage6_faiss_index_ablation": {
            "status": "out_of_scope_for_current_phase",
            "reason": "Current phase excludes FAISS Flat/IVF/HNSW ablation unless explicitly requested.",
        },
        "final_test_plan": build_final_test_plan(),
    }


def run_formal_ablation_framework() -> Dict[str, Any]:
    """Write the framework manifest and matrix artifacts under ignored results/runs."""
    ensure_data_directories()
    manifest = build_formal_ablation_manifest()
    json_path = RUNS_DIR / "formal_ablation_framework.json"
    csv_path = RUNS_DIR / "formal_ablation_matrix.csv"
    cache_path = RUNS_DIR / "formal_ablation_cache_manifest.json"
    save_json_atomic(json_path, manifest)
    write_csv(csv_path, build_formal_matrix())
    save_json_atomic(cache_path, manifest["cache_manifest"])
    manifest["artifact_paths"] = {
        "framework_json": str(json_path),
        "matrix_csv": str(csv_path),
        "cache_manifest_json": str(cache_path),
    }
    save_json_atomic(json_path, manifest)
    return manifest


def main() -> None:
    manifest = run_formal_ablation_framework()
    print("=" * 60)
    print("Formal Ablation Framework Ready")
    print("=" * 60)
    print(f"Status: {manifest['status']}")
    print(f"Dev questions: {manifest['dev_split']['question_count']}")
    print(f"Test questions: {manifest['test_split']['question_count']}")
    print(f"Matrix rows: {len(manifest['matrix'])}")
    print(f"Framework JSON: {manifest['artifact_paths']['framework_json']}")
    print(f"Matrix CSV: {manifest['artifact_paths']['matrix_csv']}")


if __name__ == "__main__":
    main()
