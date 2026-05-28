"""Runtime helpers shared by both vector stacks without importing framework adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..data.json_utils import load_json_safe


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_API_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_RERANKER_API_URL = "https://api.siliconflow.cn/v1/rerank"
DEFAULT_API_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def first_env_value(*names: str, default: str = "") -> str:
    """Return the first non-empty environment value without duplicating fallback logic."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def load_embedding_metadata(index_dir: Optional[str]) -> Dict[str, Any]:
    """Load persisted embedding metadata for a vector index when present."""
    if not index_dir:
        return {}

    metadata_path = Path(index_dir) / "build_metadata.json"
    if not metadata_path.exists():
        return {}

    try:
        return load_json_safe(metadata_path)
    except Exception as exc:
        print(f"[Embeddings] Failed to read {metadata_path}: {exc}")
        return {}


def resolve_embedding_runtime(
    index_dir: Optional[str] = None,
    *,
    default_model: str = DEFAULT_EMBEDDING_MODEL,
    model_env_var: str = "RAG_EMBEDDING_MODEL",
) -> Dict[str, Any]:
    """Resolve API embedding runtime, preferring persisted index metadata."""
    metadata = load_embedding_metadata(index_dir)
    recorded_model = metadata.get("embedding_model")
    metadata_path = str(Path(index_dir) / "build_metadata.json") if index_dir else None
    env_model = os.getenv(model_env_var)

    if recorded_model and env_model and env_model != recorded_model:
        print(
            f"[Embeddings] Ignoring {model_env_var}={env_model!r} because "
            f"index metadata records {recorded_model!r}"
        )

    return {
        "backend": "api",
        "model_name": recorded_model or env_model or default_model,
        "api_base_url": metadata.get("embedding_api_base_url")
        or first_env_value(
            "RAG_EMBEDDING_API_BASE_URL",
            default=DEFAULT_EMBEDDING_API_BASE_URL,
        ),
        "api_key": first_env_value("RAG_EMBEDDING_API_KEY", "SILICONFLOW_API_KEY"),
        "api_dimensions": metadata.get("embedding_api_dimensions"),
        "api_timeout": float(metadata.get("embedding_api_timeout", 120.0)),
        "api_max_retries": int(metadata.get("embedding_api_max_retries", 5)),
        "recorded_model": recorded_model,
        "metadata_path": metadata_path,
    }
