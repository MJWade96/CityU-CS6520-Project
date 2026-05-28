"""Guardrails for the native LlamaIndex replacement architecture."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORE_FILE = PROJECT_ROOT / "app" / "rag" / "retriever" / "vector_store.py"
EVAL_FILE = PROJECT_ROOT / "app" / "rag" / "evaluation" / "naive_rag_eval.py"
BUILD_FILE = PROJECT_ROOT / "app" / "rag" / "data" / "medical_corpus" / "build_vector_index.py"
COMPLETE_EVAL_FILE = PROJECT_ROOT / "complete_eval.py"
SAMPLE_FILE = PROJECT_ROOT / "sample_validation.py"
ENHANCED_FILE = PROJECT_ROOT / "enhanced_eval.py"
RESUME_FILE = PROJECT_ROOT / "run_with_resume.py"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
SAMPLE_IMPL_FILE = PROJECT_ROOT / "app" / "rag" / "evaluation" / "sample_validation_eval.py"


def read_text(path: Path) -> str:
    """Keep source-file assertions in one place to avoid repeated file I/O logic."""
    return path.read_text(encoding="utf-8")


def test_primary_vector_store_uses_native_llamaindex_components() -> None:
    source = read_text(STORE_FILE)

    assert "OpenAIEmbedding" in source
    assert "HuggingFaceEmbedding" not in source
    assert "FaissVectorStore" in source
    assert "VectorStoreIndex" in source


def test_primary_evaluation_uses_native_query_engine() -> None:
    source = read_text(EVAL_FILE)

    assert "OpenAILike" in source
    assert "as_query_engine" in source
    assert "run_complete_evaluation" in source


def test_primary_entrypoints_route_through_canonical_modules() -> None:
    build_source = read_text(BUILD_FILE)
    complete_source = read_text(COMPLETE_EVAL_FILE)
    sample_source = read_text(SAMPLE_FILE)

    assert "from app.rag.retriever.vector_store import MedicalVectorStore" in build_source
    assert "CHECKPOINT_FILE" in build_source
    assert "Building FAISS index" in build_source
    assert "BATCH_SIZE = 256" in build_source
    assert "INSERT_BATCH_SIZE = 8192" in build_source
    assert "USE_GPU_FAISS = False" in build_source
    assert "show_progress=True" in build_source
    assert "tqdm.write" in build_source
    assert "Indexing documents" in build_source
    assert build_source.index("load_resume_checkpoint") < build_source.index("MedicalVectorStore(")
    assert "from app.rag.evaluation.naive_rag_eval import NaiveRAGEvalConfig, run_complete_evaluation" in complete_source
    assert "from app.rag.evaluation.sample_validation_eval import SampleEvalConfig, run_sample_comparison" in sample_source


def test_parallel_llamaindex_specific_modules_are_removed() -> None:
    assert not (PROJECT_ROOT / "build_llamaindex_index.py").exists()
    assert not (PROJECT_ROOT / "llamaindex_eval.py").exists()
    assert not (PROJECT_ROOT / "llamaindex_sample_validation.py").exists()
    assert not (PROJECT_ROOT / "naive_rag_sample_eval.py").exists()
    assert not (PROJECT_ROOT / "naive_rag_retrieval.py").exists()
    assert not (PROJECT_ROOT / "naive_rag_generation.py").exists()
    assert not (PROJECT_ROOT / "naive_rag_shared.py").exists()
    assert not (
        PROJECT_ROOT / "app" / "rag" / "evaluation" / "llamaindex_rag_eval.py"
    ).exists()
    assert not (
        PROJECT_ROOT / "app" / "rag" / "retriever" / "llamaindex_store.py"
    ).exists()


def test_enhanced_surface_is_preserved() -> None:
    assert ENHANCED_FILE.exists()
    assert (PROJECT_ROOT / "app" / "rag" / "retriever" / "hybrid_retriever.py").exists()
    assert (PROJECT_ROOT / "app" / "rag" / "retriever" / "query_rewrite.py").exists()
    assert (PROJECT_ROOT / "app" / "rag" / "retriever" / "reranker.py").exists()


def test_enhanced_entrypoint_routes_through_native_module() -> None:
    source = read_text(ENHANCED_FILE)

    assert "app.rag.evaluation.enhanced_rag_eval" in source
    assert "EnhancedEvaluationConfig" in source
    assert "main" in source


def test_sample_validation_entrypoint_routes_through_native_module() -> None:
    source = read_text(SAMPLE_FILE)

    assert SAMPLE_IMPL_FILE.exists()
    assert "app.rag.evaluation.sample_validation_eval" in source
    assert "run_sample_comparison" in source


def test_primary_configs_default_to_single_faiss_store() -> None:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT.resolve()))

    from app.rag.evaluation.config import (  # pylint: disable=import-outside-toplevel
        NaiveRAGEvalConfig,
        SampleEvalConfig,
    )

    assert NaiveRAGEvalConfig().vector_store_path.name == "faiss_index"
    assert SampleEvalConfig().vector_store_path.name == "faiss_index"


def test_resume_helper_uses_primary_script_names_only() -> None:
    source = read_text(RESUME_FILE)

    assert '"complete_eval"' in source
    assert '"llamaindex_eval"' not in source


def test_resume_helper_uses_configured_results_dir() -> None:
    source = read_text(RESUME_FILE)

    assert "EVALUATION_RESULTS_DIR" in source


def test_resume_helper_defaults_to_auto_detect_mode() -> None:
    source = read_text(RESUME_FILE)

    assert "AUTO_DETECT = True" in source


def test_resolve_embedding_runtime_prefers_recorded_model_over_env(
    monkeypatch, tmp_path
) -> None:
    import json
    import sys

    sys.path.insert(0, str(PROJECT_ROOT.resolve()))

    metadata_path = tmp_path / "build_metadata.json"
    metadata_path.write_text(
        json.dumps({"embedding_model": "recorded-model"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "env-model")

    from app.rag.retriever.runtime_config import (  # pylint: disable=import-outside-toplevel
        resolve_embedding_runtime,
    )

    runtime = resolve_embedding_runtime(
        str(tmp_path),
        default_model="default-model",
    )

    assert runtime["model_name"] == "recorded-model"


def test_requirements_keep_only_native_llamaindex_packages() -> None:
    source = read_text(REQUIREMENTS_FILE)

    assert "llama-index-llms-openai-like" in source
    assert "llama-index-embeddings-huggingface" not in source
    assert "langchain" not in source.lower()


def test_native_vector_store_uses_api_embedding_and_explicit_gpu_faiss() -> None:
    source = read_text(STORE_FILE)

    assert "class BatchFaissVectorStore" in source
    assert "np.asarray" in source
    assert "run_transformations" in source
    assert "insert_nodes" in source
    assert "self.index.insert(document)" not in source
    assert "OpenAIEmbedding" in source
    assert "HuggingFaceEmbedding" not in source
    assert "use_gpu_faiss" in source
    assert "index_cpu_to_gpu" in source
    assert "index_gpu_to_cpu" in source
    assert "Install a GPU-enabled FAISS build" in source


def test_runtime_python_files_are_langchain_free() -> None:
    for path in PROJECT_ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        source = read_text(path)
        assert "langchain" not in source.lower(), path
