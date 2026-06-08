"""Generate local BGE query embeddings for the final MedQA-USMLE test split."""

from __future__ import annotations

from app.rag.data.benchmarks.medqa_usmle import load_medqa_usmle_split
from app.rag.data.data_paths import ensure_data_directories
from app.rag.experiments.formal_query_embedding_specs import QueryEmbeddingSpec
from app.rag.experiments.run_local_bge_query_embedding_autodl import (
    _load_huggingface_embedding_model,
    embed_run_queries,
)


DATASET_SPLIT = "test"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
FINAL_TEST_QUERY_SPECS = (
    QueryEmbeddingSpec(
        cache_id="final_test_naive_bge_large_k10__baai_bge-large-en-v1p5",
        pipeline="naive_rag",
        query_text_source="medqa_usmle_question_field",
    ),
    QueryEmbeddingSpec(
        cache_id=(
            "final_test_advanced_bge_large_k10_alpha0p5_rerank20"
            "__baai_bge-large-en-v1p5"
        ),
        pipeline="advanced_rag",
        query_text_source="query_rewrite_pipeline",
    ),
)


def main() -> None:
    """Reuse the formal BGE query embedding helper for the final test split."""
    import app.rag.experiments.run_local_bge_query_embedding_autodl as bge_query

    ensure_data_directories()
    bge_query.DATASET_SPLIT = DATASET_SPLIT
    questions = load_medqa_usmle_split(DATASET_SPLIT)
    embed_model = _load_huggingface_embedding_model(EMBEDDING_MODEL)
    for spec in FINAL_TEST_QUERY_SPECS:
        embed_run_queries(
            spec,
            questions,
            embedding_model=EMBEDDING_MODEL,
            embed_model=embed_model,
        )


if __name__ == "__main__":
    main()
