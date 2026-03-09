"""
RAG Module for Medical Diagnosis
LangChain-based Retrieval-Augmented Generation System

Architecture:
- LLM ONLY used in Generator Module (Qwen2.5-7B)
- All other modules use local methods (NO LLM)
"""

from .config import RAGConfig, DEFAULT_CONFIG
from .document_loader import (
    MedicalDocumentLoader,
    MedicalKnowledgeBase,
    load_sample_knowledge_base,
    SAMPLE_MEDICAL_KNOWLEDGE
)
from .embeddings import (
    BaseEmbeddingModel,
    HuggingFaceEmbeddingModel,
    MockEmbeddingModel,
    get_embedding_model,
    get_langchain_embeddings
)
from .vector_store import MedicalVectorStore, VectorStoreManager
from .pipeline import (
    MedicalRAGPipeline,
    RetrievalResult,
    RAGResult,
    create_rag_pipeline
)

# Qwen2.5-7B Version - LLM ONLY in Generator
try:
    from .qwen_medical_rag import (
        MedicalRAGSystem,
        MedicalRAGConfig as QwenConfig,
        LocalEmbeddingModel,
        QwenGenerator,
        RuleBasedEvaluator,
        create_rag_system
    )
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

__all__ = [
    # Config
    'RAGConfig',
    'DEFAULT_CONFIG',
    
    # Document Loader
    'MedicalDocumentLoader',
    'MedicalKnowledgeBase',
    'load_sample_knowledge_base',
    'SAMPLE_MEDICAL_KNOWLEDGE',
    
    # Embeddings
    'BaseEmbeddingModel',
    'HuggingFaceEmbeddingModel',
    'MockEmbeddingModel',
    'get_embedding_model',
    'get_langchain_embeddings',
    
    # Vector Store
    'MedicalVectorStore',
    'VectorStoreManager',
    
    # Pipeline
    'MedicalRAGPipeline',
    'RetrievalResult',
    'RAGResult',
    'create_rag_pipeline',
]

# Add Qwen modules if available
if QWEN_AVAILABLE:
    __all__.extend([
        'MedicalRAGSystem',
        'QwenConfig',
        'LocalEmbeddingModel',
        'QwenGenerator',
        'RuleBasedEvaluator',
        'create_rag_system',
    ])
