"""
Embeddings Module
Handles text embedding generation using LangChain and various embedding providers
"""

import os
from typing import List, Optional
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
        'default': 'sentence-transformers/all-MiniLM-L6-v2',
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
    model_type: str = "huggingface",
    **kwargs
) -> BaseEmbeddingModel:
    """
    Factory function to get an embedding model.
    
    Args:
        model_type: Type of embedding model ('huggingface', 'openai', 'mock')
        **kwargs: Additional arguments passed to the model constructor
        
    Returns:
        BaseEmbeddingModel instance
    """
    if model_type == "huggingface":
        return HuggingFaceEmbeddingModel(**kwargs)
    elif model_type == "openai":
        return OpenAIEmbeddingModel(**kwargs)
    elif model_type == "mock":
        return MockEmbeddingModel(**kwargs)
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")


# Convenience function for LangChain compatibility
def get_langchain_embeddings(
    model_type: str = "huggingface",
    **kwargs
) -> Embeddings:
    """
    Get a LangChain-compatible embeddings instance.
    
    Args:
        model_type: Type of embedding model
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
