"""Primary vector store and index builder contracts."""

from __future__ import annotations

import inspect


def test_primary_vector_store_exposes_native_faiss_runtime_contract() -> None:
    from app.rag.retriever.vector_store import BatchFaissVectorStore, MedicalVectorStore

    constructor = inspect.signature(MedicalVectorStore)
    expected_settings = {
        "embedding_model_name",
        "embedding_api_base_url",
        "embedding_api_key",
        "embedding_api_num_workers",
        "index_use_async",
        "use_gpu_faiss",
    }

    assert expected_settings.issubset(constructor.parameters)
    assert issubclass(BatchFaissVectorStore, object)
    for method_name in (
        "build",
        "add_documents",
        "as_query_engine",
        "retrieve",
        "similarity_search_with_score",
        "save",
        "load",
    ):
        assert callable(getattr(MedicalVectorStore, method_name))


def test_index_builder_defaults_capture_resume_and_async_contract() -> None:
    from app.rag.data.medical_corpus import build_vector_index as module

    config = module.DEFAULT_INDEX_BUILD_CONFIG
    checkpoint = module.checkpoint_payload(
        completed_documents=3,
        total_documents=10,
        elapsed=1.25,
        config=config,
    )

    assert config.batch_size == 64
    assert config.embedding_api_num_workers == 4
    assert config.index_use_async is True
    assert config.insert_batch_size == 8192
    assert config.use_gpu_faiss is False
    assert config.faiss_index_type == "FlatIP"
    assert checkpoint["completed_documents"] == 3
    assert checkpoint["embedding_backend"] == "api"
    assert checkpoint["index_use_async"] is True
