"""Index builder checkpoint consumability."""

from __future__ import annotations


def test_checkpoint_payload_records_resume_consumable_fields() -> None:
    from app.rag.data.medical_corpus import build_vector_index as module

    config = module.DEFAULT_INDEX_BUILD_CONFIG
    checkpoint = module.checkpoint_payload(
        completed_documents=3,
        total_documents=10,
        elapsed=1.25,
        config=config,
    )

    assert checkpoint["completed_documents"] == 3
    assert checkpoint["embedding_backend"] == "api"
    assert checkpoint["index_use_async"] is True
