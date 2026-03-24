"""
RAG Configuration
Configuration settings for the Medical Diagnosis RAG System
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

from .data_paths import MEDICAL_KNOWLEDGE_DIR, VECTOR_STORE_DIR

class EmbeddingModelType(str, Enum):
    """Supported embedding model types"""

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

    ZHIPU = "zhipu"
    LOCAL = "local"


@dataclass
class RAGConfig:
    """RAG System Configuration"""

    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])

    embedding_model: EmbeddingModelType = EmbeddingModelType.BGE
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    vector_store: VectorStoreType = VectorStoreType.FAISS
    persist_directory: Optional[str] = str(VECTOR_STORE_DIR)
    top_k: int = 5

    llm_type: LLMType = LLMType.ZHIPU
    llm_model: str = "glm-4"
    temperature: float = 0.3
    max_tokens: int = 2048

    knowledge_base_path: Optional[str] = str(
        MEDICAL_KNOWLEDGE_DIR / "medical_knowledge.json"
    )
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# Default configuration instance
DEFAULT_CONFIG = RAGConfig()
