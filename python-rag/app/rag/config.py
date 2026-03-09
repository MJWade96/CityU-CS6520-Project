"""
RAG Configuration
Configuration settings for the Medical Diagnosis RAG System
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class EmbeddingModelType(str, Enum):
    """Supported embedding model types"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMER = "sentence-transformer"
    BGE = "bge"


class VectorStoreType(str, Enum):
    """Supported vector store types"""
    FAISS = "faiss"
    CHROMA = "chroma"
    IN_MEMORY = "in_memory"


class LLMType(str, Enum):
    """Supported LLM types"""
    OPENAI = "openai"
    ZHIPU = "zhipu"
    LOCAL = "local"


@dataclass
class RAGConfig:
    """RAG System Configuration"""
    
    # Document Processing
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    
    # Embedding Settings
    embedding_model: EmbeddingModelType = EmbeddingModelType.BGE
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    
    # Vector Store Settings
    vector_store: VectorStoreType = VectorStoreType.FAISS
    persist_directory: Optional[str] = "./data/vector_store"
    
    # Retrieval Settings
    top_k: int = 5
    
    # LLM Settings
    llm_type: LLMType = LLMType.ZHIPU
    llm_model: str = "glm-4"
    temperature: float = 0.3
    max_tokens: int = 2048
    
    # Medical Knowledge Base
    knowledge_base_path: str = "./data/medical_knowledge"
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# Default configuration instance
DEFAULT_CONFIG = RAGConfig()
