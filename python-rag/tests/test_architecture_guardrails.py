"""Minimal architecture guardrails that are intentionally centralized."""

from __future__ import annotations

import inspect
from pathlib import Path

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


def test_experiment_entrypoints_are_owned_by_experiments_package() -> None:
    for script_name in ROOT_ENTRYPOINT_NAMES:
        assert not (PROJECT_ROOT / script_name).exists()
        assert (EXPERIMENTS_DIR / script_name).exists()


def test_formal_entrypoint_is_module_execution_surface() -> None:
    from app.rag.experiments import run_formal_ablation

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "python -m app.rag.experiments.run_formal_ablation" in readme
    assert inspect.signature(run_formal_ablation.main).parameters == {}


def test_local_embeddings_stay_outside_primary_vector_store() -> None:
    from app.rag.experiments import phase1_formal_ablation as module
    from app.rag.retriever.vector_store import MedicalVectorStore

    assert any(
        provider.backend in module.LOCAL_EMBEDDING_BACKENDS
        for provider in module.EMBEDDING_PROVIDERS
    )
    assert "embedding_backend" not in inspect.signature(MedicalVectorStore).parameters
    assert (
        PROJECT_ROOT / "app" / "rag" / "evaluation" / "formal_local_embedding_adapter.py"
    ).exists()


def test_duplicate_formal_runtime_is_not_a_supported_surface() -> None:
    assert not (
        PROJECT_ROOT / "app" / "rag" / "experiments" / "formal_ablation_runtime.py"
    ).exists()


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
