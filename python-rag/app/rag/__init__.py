"""
RAG Module for Medical Diagnosis
LangChain-based Retrieval-Augmented Generation System

Architecture:
- LLM ONLY used in Generator Module (API)
- All other modules use local methods (NO LLM)
"""

from .config import RAGConfig, DEFAULT_CONFIG
from .document_loader import (
    MedicalDocumentLoader,
    MedicalKnowledgeBase,
    load_sample_knowledge_base,
    SAMPLE_MEDICAL_KNOWLEDGE,
)
from .embeddings import (
    BaseEmbeddingModel,
    HuggingFaceEmbeddingModel,
    MockEmbeddingModel,
    get_embedding_model,
    get_langchain_embeddings,
)
from .vector_store import MedicalVectorStore, VectorStoreManager
from .api_medical_rag import (
    MedicalRAGSystem,
    MedicalRAGConfig,
    APIGenerator,
    RuleBasedEvaluator,
    create_rag_system,
    create_rag_pipeline,
)

__all__ = [
    # Config
    "RAGConfig",
    "DEFAULT_CONFIG",
    # Document Loader
    "MedicalDocumentLoader",
    "MedicalKnowledgeBase",
    "load_sample_knowledge_base",
    "SAMPLE_MEDICAL_KNOWLEDGE",
    # Embeddings
    "BaseEmbeddingModel",
    "HuggingFaceEmbeddingModel",
    "MockEmbeddingModel",
    "get_embedding_model",
    "get_langchain_embeddings",
    # Vector Store
    "MedicalVectorStore",
    "VectorStoreManager",
    # RAG System
    "MedicalRAGSystem",
    "MedicalRAGConfig",
    "APIGenerator",
    "RuleBasedEvaluator",
    "create_rag_system",
    "create_rag_pipeline",
]
