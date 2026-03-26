"""
Embeddings Module
Handles text embedding generation using LangChain and various embedding providers
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings,
)

# Try to import OpenAI embeddings if available
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


DEFAULT_HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
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
        return json.loads(metadata_path.read_text(encoding="utf-8"))
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
        'default': DEFAULT_HF_EMBEDDING_MODEL,
        'medical': 'emilyalsentzer/Bio_ClinicalBERT',
        'pubmed': 'sentence-transformers/all-mpnet-base-v2',
        'biobert': 'dmis-lab/biobert-base-cased-v1.1',
    }
    
    def __init__(
        self,
        model_name: str = 'default',
        model_kwargs: Optional[dict] = None,
        encode_kwargs: Optional[dict] = None
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
        self.model_kwargs = model_kwargs or {'device': 'cpu'}
        self.encode_kwargs = encode_kwargs or {'normalize_embeddings': True}
        
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


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    OpenAI embedding model
    
    Uses OpenAI's text-embedding models for high-quality embeddings.
    Requires OPENAI_API_KEY environment variable.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI embedding model.
        
        Args:
            model: OpenAI embedding model name
            openai_api_key: OpenAI API key (defaults to env variable)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("langchain-openai is required for OpenAI embeddings")
        
        self.model = model
        
        # Get API key
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key
        )
        
        # Dimension mapping for OpenAI models
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embeddings.embed_query(text)
    
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self._dimensions.get(self.model, 1536)


class MockEmbeddingModel(BaseEmbeddingModel):
    """
    Mock embedding model for testing and development
    
    Generates deterministic embeddings based on text hashing.
    Not suitable for production use.
    """
    
    def __init__(self, dimension: int = 384):
        """Initialize mock embedding model"""
        self.dimension = dimension
    
    def _hash_text(self, text: str) -> List[float]:
        """Generate a deterministic embedding from text hash"""
        import hashlib
        import math
        
        # Create hash of text
        hash_bytes = hashlib.sha256(text.encode()).digest()
        
        # Convert to float vector
        embedding = []
        for i in range(self.dimension):
            # Use modulo to cycle through hash bytes
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Normalize to [-1, 1] range
            val = (byte_val / 128.0) - 1.0
            embedding.append(val)
        
        # Normalize the vector
        magnitude = math.sqrt(sum(v * v for v in embedding))
        if magnitude > 0:
            embedding = [v / magnitude for v in embedding]
        
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return [self._hash_text(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._hash_text(text)
    
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.dimension


def get_embedding_model(
    model_type: str = "bge",
    **kwargs
) -> BaseEmbeddingModel:
    """
    Factory function to get an embedding model.

    Args:
        model_type: Type of embedding model ('bge', 'huggingface', 'openai', 'mock')
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        BaseEmbeddingModel instance
    """
    if model_type == "bge":
        model_name = kwargs.get('model_name', 'BAAI/bge-small-en-v1.5')
        return HuggingFaceEmbeddingModel(model_name=model_name, **kwargs)
    elif model_type == "huggingface":
        return HuggingFaceEmbeddingModel(**kwargs)
    elif model_type == "openai":
        return OpenAIEmbeddingModel(**kwargs)
    elif model_type == "mock":
        return MockEmbeddingModel(**kwargs)
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")


# Convenience function for LangChain compatibility
def get_langchain_embeddings(
    model_type: str = "bge",
    **kwargs
) -> Embeddings:
    """
    Get a LangChain-compatible embeddings instance.

    Args:
        model_type: Type of embedding model ('bge', 'huggingface', 'openai', 'mock')
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


if __name__ == "__main__":
    # Test embedding models
    print("Testing embedding models...")
    
    # Test mock model
    mock_model = MockEmbeddingModel()
    embedding = mock_model.embed_query("test query")
    print(f"Mock embedding dimension: {len(embedding)}")
    
    # Test HuggingFace model (if available)
    try:
        hf_model = HuggingFaceEmbeddingModel(model_name='default')
        print(f"HuggingFace model dimension: {hf_model.get_dimension()}")
    except Exception as e:
        print(f"HuggingFace model not available: {e}")
