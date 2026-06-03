"""Minimal architecture guardrails that are intentionally centralized."""

from __future__ import annotations

import importlib
import inspect

from conftest import PROJECT_ROOT


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


def test_experiment_entrypoints_expose_module_execution_surfaces() -> None:
    for script_name in ROOT_ENTRYPOINT_NAMES:
        module_name = script_name.removesuffix(".py")
        module = importlib.import_module(f"app.rag.experiments.{module_name}")

        assert (EXPERIMENTS_DIR / script_name).samefile(module.__file__)
        assert callable(module.main)
        assert inspect.signature(module.main).parameters == {}


def test_formal_entrypoint_is_module_execution_surface() -> None:
    from app.rag.experiments import run_formal_ablation

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "python -m app.rag.experiments.run_formal_ablation" in readme
    assert inspect.signature(run_formal_ablation.main).parameters == {}


def test_local_embeddings_stay_outside_primary_vector_store() -> None:
    from app.rag.experiments import phase1_formal_ablation as module
    from app.rag.evaluation.formal_local_embedding_adapter import (
        LocalEmbeddingFormalRetriever,
    )
    from app.rag.retriever.vector_store import MedicalVectorStore

    vector_store_params = set(inspect.signature(MedicalVectorStore).parameters)

    assert any(
        provider.backend in module.LOCAL_EMBEDDING_BACKENDS
        for provider in module.EMBEDDING_PROVIDERS
    )
    assert {
        "embedding_model_name",
        "embedding_api_base_url",
        "embedding_api_key",
    }.issubset(vector_store_params)
    assert callable(LocalEmbeddingFormalRetriever.load)
    assert callable(LocalEmbeddingFormalRetriever.retrieve)
    assert callable(LocalEmbeddingFormalRetriever.retrieve_components)


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
