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
    load_medical_knowledge_base,
    load_knowledge_from_file,
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
from .corpus_loader import MedicalCorpusLoader, StatPearlsLoader, PubMedLoader
from .corpus_registry import CORPUS_REGISTRY, combine_registered_corpora
from .statpearls_dataset import build_statpearls_dataset
from .prompt_template import (
    MedicalPromptManager,
    create_rag_prompt_template,
    create_chat_prompt_template,
    format_context,
    format_options,
    create_rag_inputs,
    MEDICAL_RAG_PROMPT,
    SYSTEM_MESSAGE,
)
from .progress_manager import (
    EvaluationProgressManager,
    create_progress_manager,
    CheckpointData,
)

__all__ = [
    # Config
    "RAGConfig",
    "DEFAULT_CONFIG",
    # Document Loader
    "MedicalDocumentLoader",
    "MedicalKnowledgeBase",
    "load_sample_knowledge_base",
    "load_medical_knowledge_base",
    "load_knowledge_from_file",
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
    # Corpus Loader
    "MedicalCorpusLoader",
    "StatPearlsLoader",
    "PubMedLoader",
    "CORPUS_REGISTRY",
    "combine_registered_corpora",
    "build_statpearls_dataset",
    # Prompt Templates
    "MedicalPromptManager",
    "create_rag_prompt_template",
    "create_chat_prompt_template",
    "format_context",
    "format_options",
    "create_rag_inputs",
    "MEDICAL_RAG_PROMPT",
    "SYSTEM_MESSAGE",
    # Progress Manager
    "EvaluationProgressManager",
    "create_progress_manager",
    "CheckpointData",
]
