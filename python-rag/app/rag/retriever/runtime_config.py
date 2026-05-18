"""Runtime helpers shared by both vector stacks without importing framework adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..data.json_utils import load_json_safe


DEFAULT_HF_EMBEDDING_MODEL = "BAAI/bge-m3"


def _is_torch_device_available(device: str) -> bool:
    """Check whether the requested torch device can be used."""
    if device == "cpu":
        return True

    try:
        import torch
    except Exception:
        return False

    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    return False


def resolve_torch_device(
    preferred_device: Optional[str] = None,
    *,
    env_var: Optional[str] = "RAG_EMBEDDING_DEVICE",
) -> str:
    """Resolve a torch device with automatic fallback when accelerators are unavailable."""
    raw_value = preferred_device
    if raw_value is None and env_var:
        raw_value = os.getenv(env_var)

    requested = (raw_value or "auto").strip().lower()

    if requested == "auto":
        for candidate in ("cuda", "mps"):
            if _is_torch_device_available(candidate):
                return candidate
        return "cpu"

    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")

    if _is_torch_device_available(requested):
        return requested

    print(f"[Torch] Requested device '{requested}' is unavailable; falling back to CPU")
    return "cpu"


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
    default_model: str = DEFAULT_HF_EMBEDDING_MODEL,
    preferred_device: Optional[str] = None,
    model_env_var: str = "RAG_EMBEDDING_MODEL",
    device_env_var: str = "RAG_EMBEDDING_DEVICE",
) -> Dict[str, Any]:
    """Resolve the runtime embedding model/device, preferring persisted index metadata."""
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
        "model_name": recorded_model or env_model or default_model,
        "device": resolve_torch_device(preferred_device, env_var=device_env_var),
        "recorded_model": recorded_model,
        "metadata_path": metadata_path,
    }