"""
Embeddings Module
Handles text embedding generation for the embedding backends used by this project
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from .json_utils import load_json_safe


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

    return {
        "model_name": env_model or recorded_model or default_model,
        "device": resolve_torch_device(preferred_device, env_var=device_env_var),
        "recorded_model": recorded_model,
        "metadata_path": metadata_path,
    }


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        pass


class HuggingFaceEmbeddingModel(BaseEmbeddingModel):
    """
    HuggingFace embedding model using sentence-transformers

    Uses medical-domain specific models when available for better
    performance on medical text.
    """

    # Recommended models for medical text
    MEDICAL_MODELS = {
        "default": DEFAULT_HF_EMBEDDING_MODEL,
        "medical": "emilyalsentzer/Bio_ClinicalBERT",
        "pubmed": "sentence-transformers/all-mpnet-base-v2",
        "biobert": "dmis-lab/biobert-base-cased-v1.1",
        "bge-m3": "BAAI/bge-m3",
        "bge-small": "BAAI/bge-small-en-v1.5",
    }

    def __init__(
        self,
        model_name: str = "default",
        model_kwargs: Optional[dict] = None,
        encode_kwargs: Optional[dict] = None,
    ):
        """
        Initialize HuggingFace embedding model.

        Args:
            model_name: Name of the model or key from MEDICAL_MODELS
            model_kwargs: Additional arguments for model initialization
            encode_kwargs: Additional arguments for encoding
        """
        # Resolve model name
        if model_name in self.MEDICAL_MODELS:
            model_name = self.MEDICAL_MODELS[model_name]

        self.model_name = model_name

        # Default configurations
        self.model_kwargs = model_kwargs or {"device": "cpu"}
        self.encode_kwargs = encode_kwargs or {"normalize_embeddings": True}

        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs,
        )

        # Cache dimension
        self._dimension = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embeddings.embed_query(text)

    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        if self._dimension is None:
            # Get dimension by embedding a sample text
            sample = self.embed_query("sample")
            self._dimension = len(sample)
        return self._dimension


def get_embedding_model(model_type: str = "bge-m3", **kwargs) -> BaseEmbeddingModel:
    """
    Factory function to get an embedding model.

    Args:
        model_type: Type of embedding model ('bge-m3', 'bge', 'huggingface')
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        BaseEmbeddingModel instance
    """
    if model_type == "bge-m3":
        model_name = kwargs.pop("model_name", "BAAI/bge-m3")
        return HuggingFaceEmbeddingModel(model_name=model_name, **kwargs)
    elif model_type == "bge":
        model_name = kwargs.pop("model_name", "BAAI/bge-small-en-v1.5")
        return HuggingFaceEmbeddingModel(model_name=model_name, **kwargs)
    elif model_type == "huggingface":
        return HuggingFaceEmbeddingModel(**kwargs)
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")


# Convenience function for LangChain compatibility
def get_langchain_embeddings(
    model_type: str = "bge-m3",
    **kwargs
) -> Embeddings:
    """
    Get a LangChain-compatible embeddings instance.

    Args:
        model_type: Type of embedding model ('bge-m3', 'bge', 'huggingface')
        **kwargs: Additional arguments

    Returns:
        LangChain Embeddings instance
    """
    model = get_embedding_model(model_type, **kwargs)

    # Wrap in LangChain-compatible interface
    class LangChainEmbeddingsWrapper(Embeddings):
        def __init__(self, embedding_model):
            self.model = embedding_model

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self.model.embed_documents(texts)

        def embed_query(self, text: str) -> List[float]:
            return self.model.embed_query(text)

    return LangChainEmbeddingsWrapper(model)
