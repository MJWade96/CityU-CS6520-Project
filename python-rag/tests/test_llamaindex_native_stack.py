"""Focused guardrails for the native LlamaIndex runtime boundaries."""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.resolve()))

EXPERIMENTS_DIR = PROJECT_ROOT / "app" / "rag" / "experiments"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
ROOT_ENTRYPOINT_NAMES = {
    "complete_eval.py",
    "enhanced_eval.py",
    "evaluate_no_rag.py",
    "run_formal_ablation.py",
    "run_with_resume.py",
    "sample_validation.py",
}


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


def test_experiment_entrypoints_are_owned_by_experiments_package() -> None:
    for script_name in ROOT_ENTRYPOINT_NAMES:
        assert not (PROJECT_ROOT / script_name).exists()
        assert (EXPERIMENTS_DIR / script_name).exists()


def test_formal_entrypoint_is_module_execution_surface() -> None:
    from app.rag.experiments import run_formal_ablation

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "python -m app.rag.experiments.run_formal_ablation" in readme
    assert inspect.signature(run_formal_ablation.main).parameters == {}


def test_primary_config_defaults_point_to_shared_faiss_store() -> None:
    from app.rag.evaluation.config import NaiveRAGEvalConfig, SampleEvalConfig

    assert NaiveRAGEvalConfig().vector_store_path.name == "faiss_index"
    assert SampleEvalConfig().vector_store_path.name == "faiss_index"


def test_openai_like_kwargs_keep_enable_thinking_inside_extra_body() -> None:
    from llama_index.llms.openai_like import OpenAILike

    from app.rag.evaluation.eval_shared import (
        EvaluationLLMConfig,
        get_qwen_openai_like_kwargs,
    )

    kwargs = get_qwen_openai_like_kwargs(EvaluationLLMConfig(enable_thinking=False))
    model_kwargs = OpenAILike(**kwargs)._get_model_kwargs(max_tokens=7)

    assert kwargs["additional_kwargs"] == {"extra_body": {"enable_thinking": False}}
    assert "enable_thinking" not in kwargs["additional_kwargs"]
    assert model_kwargs["extra_body"] == {"enable_thinking": False}
    assert "enable_thinking" not in model_kwargs


def test_llm_defaults_use_ctyun_generator_with_requested_limits() -> None:
    from app.rag.evaluation.eval_shared import ConcurrencyConfig, EvaluationLLMConfig

    llm = EvaluationLLMConfig()
    concurrency = ConcurrencyConfig()

    assert llm.provider == "ctyun"
    assert llm.model == "8606056bfe0c49448d92587452d1f2fc"
    assert llm.base_url == "https://wishub-x6.ctyun.cn/v1/"
    assert llm.api_key
    assert llm.temperature == 0.1
    assert llm.enable_thinking is True
    assert concurrency.rpm_limit == 1000
    assert concurrency.tpm_limit == 50000


def test_llm_clients_ignore_environment_proxies_by_default() -> None:
    from app.rag.evaluation.eval_shared import (
        EvaluationLLMConfig,
        create_async_client,
        get_qwen_openai_like_kwargs,
    )

    config = EvaluationLLMConfig(api_key="test-key")
    client = create_async_client(config)
    kwargs = get_qwen_openai_like_kwargs(config)

    try:
        assert getattr(client._client, "_trust_env") is False
        assert getattr(kwargs["http_client"], "_trust_env") is False
        assert getattr(kwargs["async_http_client"], "_trust_env") is False
    finally:
        asyncio.run(client.close())
        kwargs["http_client"].close()
        asyncio.run(kwargs["async_http_client"].aclose())


def test_llm_config_reads_environment_at_instantiation(monkeypatch) -> None:
    from app.rag.evaluation.eval_shared import EvaluationLLMConfig

    monkeypatch.setenv("RAG_LLM_MODEL", "env-model")
    monkeypatch.setenv("RAG_LLM_API_KEY", "env-key")

    llm = EvaluationLLMConfig()

    assert llm.model == "env-model"
    assert llm.api_key == "env-key"


def test_resume_helper_keeps_primary_runtime_names() -> None:
    from app.rag.experiments import run_with_resume

    assert run_with_resume.AUTO_DETECT is True
    assert "complete_eval" in run_with_resume.get_checkpoint_script_names("complete_eval")
    assert run_with_resume.get_script_path("complete_eval").name == "complete_eval.py"
    assert run_with_resume.get_checkpoint_script_names("llamaindex_eval") == [
        "llamaindex_eval"
    ]
    assert str(run_with_resume.EVALUATION_RESULTS_DIR).endswith("evaluation")


def test_resolve_embedding_runtime_prefers_recorded_model_over_env(
    monkeypatch, tmp_path
) -> None:
    import json

    metadata_path = tmp_path / "build_metadata.json"
    metadata_path.write_text(
        json.dumps({"embedding_model": "recorded-model"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "env-model")

    from app.rag.retriever.runtime_config import resolve_embedding_runtime

    runtime = resolve_embedding_runtime(
        str(tmp_path),
        default_model="default-model",
    )

    assert runtime["model_name"] == "recorded-model"


def test_requirements_keep_native_llamaindex_dependency_boundary() -> None:
    def package_name(line: str) -> str:
        return (
            line.split("#", maxsplit=1)[0]
            .strip()
            .split("==", maxsplit=1)[0]
            .split(">=", maxsplit=1)[0]
            .split("<=", maxsplit=1)[0]
            .lower()
        )

    requirements = {
        package_name(line)
        for line in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert "llama-index-llms-openai-like" in requirements
    assert "llama-index-embeddings-huggingface" in requirements
    assert "langchain" not in requirements
    assert "langchain-community" not in requirements


def test_runtime_python_files_keep_langchain_out_of_primary_runtime() -> None:
    allowed = {Path("tests"), Path("__pycache__")}
    for path in PROJECT_ROOT.rglob("*.py"):
        relative = path.relative_to(PROJECT_ROOT)
        if any(relative.parts[:1] == folder.parts for folder in allowed):
            continue
        source = path.read_text(encoding="utf-8")
        assert "langchain" not in source.lower(), path
