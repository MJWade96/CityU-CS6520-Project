"""
RAG Pipeline Module
Main Retrieval-Augmented Generation pipeline using LangChain
"""

import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings

# LangChain LLM imports
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_community.chat_models import ChatZhipuAI
    ZHIPU_AVAILABLE = True
except ImportError:
    ZHIPU_AVAILABLE = False

from langchain_community.llms import FakeListLLM

from .config import RAGConfig, DEFAULT_CONFIG
from .document_loader import MedicalKnowledgeBase, load_sample_knowledge_base
from .embeddings import get_langchain_embeddings
from .vector_store import MedicalVectorStore


@dataclass
class RetrievalResult:
    """Result from the retrieval stage"""
    query: str
    documents: List[Document]
    scores: List[float]
    retrieval_time: float


@dataclass
class RAGResult:
    """Complete RAG pipeline result"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    retrieval_result: RetrievalResult
    total_time: float


# Medical RAG Prompt Template
MEDICAL_RAG_PROMPT = PromptTemplate.from_template(
    """You are a medical diagnosis assistant powered by RAG (Retrieval-Augmented Generation).
Your role is to provide accurate, evidence-based medical information based on the retrieved context.

IMPORTANT GUIDELINES:
1. Base your answers primarily on the provided context from medical literature
2. Always cite the sources when providing information
3. If the context does not contain sufficient information, clearly state this
4. Provide balanced information including benefits, risks, and alternatives when relevant
5. Use clear, accessible language while maintaining medical accuracy
6. Always recommend consulting healthcare professionals for specific medical advice

CONTEXT FROM MEDICAL LITERATURE:
{context}

QUESTION: {question}

Please provide a comprehensive answer based on the context above."""
)


class MedicalRAGPipeline:
    """
    Medical RAG Pipeline using LangChain
    
    Implements a complete RAG pipeline for medical diagnosis support,
    including document retrieval, reranking, and answer generation.
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedding_model: Optional[Embeddings] = None,
        knowledge_base: Optional[MedicalKnowledgeBase] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: RAG configuration
            embedding_model: LangChain embeddings instance
            knowledge_base: Pre-loaded medical knowledge base
        """
        self.config = config or DEFAULT_CONFIG
        self.is_initialized = False
        
        # Initialize embedding model
        if embedding_model is None:
            self.embedding_model = get_langchain_embeddings(model_type="mock")
        else:
            self.embedding_model = embedding_model
        
        # Initialize vector store
        self.vector_store = MedicalVectorStore(
            embedding_model=self.embedding_model,
            store_type="faiss"
        )
        
        # Initialize knowledge base
        self.knowledge_base = knowledge_base
        
        # Initialize LLM (will be set during initialization)
        self.llm = None
    
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the pipeline with medical knowledge.
        
        Returns:
            Dictionary with initialization statistics
        """
        start_time = time.time()
        
        # Load knowledge base if not provided
        if self.knowledge_base is None:
            self.knowledge_base = load_sample_knowledge_base()
        
        # Get documents from knowledge base
        documents = self.knowledge_base.get_documents()
        
        # Add documents to vector store
        self.vector_store.add_documents(documents)
        
        # Initialize LLM
        self._initialize_llm()
        
        self.is_initialized = True
        
        return {
            'total_documents': len(documents),
            'initialization_time': time.time() - start_time,
            'vector_store_stats': self.vector_store.get_stats()
        }
    
    def _initialize_llm(self):
        """Initialize the language model"""
        # Try ZhipuAI first (for Chinese environment)
        if ZHIPU_AVAILABLE:
            try:
                import os
                api_key = os.getenv("ZHIPUAI_API_KEY")
                if api_key:
                    self.llm = ChatZhipuAI(
                        model=self.config.llm_model,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                    print("Initialized ZhipuAI LLM")
                    return
            except Exception as e:
                print(f"ZhipuAI initialization failed: {e}")
        
        # Try OpenAI
        if OPENAI_AVAILABLE:
            try:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                    print("Initialized OpenAI LLM")
                    return
            except Exception as e:
                print(f"OpenAI initialization failed: {e}")
        
        # Fallback to mock LLM
        self.llm = FakeListLLM(
            responses=[
                "Based on the provided medical literature, I can provide information on this topic. "
                "Please consult a healthcare professional for specific medical advice. "
                "The information provided is for educational purposes only."
            ]
        )
        print("Initialized Mock LLM (for testing)")
    
    def query(self, question: str) -> RAGResult:
        """
        Process a medical query through the RAG pipeline.
        
        Args:
            question: Medical question to answer
            
        Returns:
            RAGResult with answer and metadata
        """
        if not self.is_initialized:
            self.initialize()
        
        total_start_time = time.time()
        
        # Step 1: Retrieve relevant documents
        retrieval_result = self.retrieve(question)
        
        # Step 2: Generate answer
        answer = self.generate(question, retrieval_result)
        
        # Step 3: Build sources
        sources = self._build_sources(
            retrieval_result.documents,
            retrieval_result.scores
        )
        
        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(retrieval_result)
        
        total_time = time.time() - total_start_time
        
        return RAGResult(
            query=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            retrieval_result=retrieval_result,
            total_time=total_time
        )
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            RetrievalResult with documents and scores
        """
        start_time = time.time()
        
        k = k or self.config.top_k
        
        # Perform similarity search with scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Separate documents and scores
        documents = [doc for doc, score in results]
        scores = [float(score) for doc, score in results]
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(
            query=query,
            documents=documents,
            scores=scores,
            retrieval_time=retrieval_time
        )
    
    def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult
    ) -> str:
        """
        Generate an answer using retrieved documents.
        
        Args:
            query: Original query
            retrieval_result: Retrieved documents
            
        Returns:
            Generated answer string
        """
        # Format context from retrieved documents
        context = self._format_context(retrieval_result.documents)
        
        # Create the prompt
        prompt_value = MEDICAL_RAG_PROMPT.format(
            context=context,
            question=query
        )
        
        # Generate answer using LLM
        if self.llm:
            try:
                response = self.llm.invoke(prompt_value)
                # Handle different response types
                if hasattr(response, 'content'):
                    return response.content
                return str(response)
            except Exception as e:
                print(f"LLM generation error: {e}")
                return self._generate_fallback_answer(query, retrieval_result)
        else:
            return self._generate_fallback_answer(query, retrieval_result)
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return "No relevant medical information found in the knowledge base."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            title = doc.metadata.get('title', 'Unknown')
            context_parts.append(
                f"[{i}] Source: {source}\n"
                f"Title: {title}\n"
                f"{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_fallback_answer(
        self,
        query: str,
        retrieval_result: RetrievalResult
    ) -> str:
        """Generate a fallback answer when LLM is not available"""
        if not retrieval_result.documents:
            return (
                "I apologize, but I could not find relevant medical information "
                "for your query in the knowledge base. Please try rephrasing your "
                "question or consult a healthcare professional for medical advice."
            )
        
        # Build answer from retrieved content
        answer_parts = [
            "Based on the retrieved medical literature, here is the relevant information:\n"
        ]
        
        for i, doc in enumerate(retrieval_result.documents, 1):
            title = doc.metadata.get('title', 'Unknown')
            source = doc.metadata.get('source', 'Unknown')
            answer_parts.append(f"\n{i}. From {title} ({source}):")
            answer_parts.append(f"   {doc.page_content[:300]}...")
        
        answer_parts.append(
            "\n\nPlease note: This information is for educational purposes. "
            "Always consult healthcare professionals for medical advice."
        )
        
        return "\n".join(answer_parts)
    
    def _build_sources(
        self,
        documents: List[Document],
        scores: List[float]
    ) -> List[Dict[str, Any]]:
        """Build source list from retrieved documents"""
        sources = []
        
        for doc, score in zip(documents, scores):
            sources.append({
                'title': doc.metadata.get('title', 'Unknown'),
                'source': doc.metadata.get('source', 'Unknown'),
                'category': doc.metadata.get('category', 'general'),
                'relevance_score': float(score),
                'chunk_index': doc.metadata.get('chunk_index'),
            })
        
        return sources
    
    def _calculate_confidence(self, retrieval_result: RetrievalResult) -> float:
        """Calculate confidence score based on retrieval results"""
        if not retrieval_result.scores:
            return 0.0
        
        # Normalize scores (FAISS uses distance, so invert)
        normalized_scores = [1.0 / (1.0 + s) for s in retrieval_result.scores]
        
        # Average score
        avg_score = sum(normalized_scores) / len(normalized_scores)
        
        # Number of documents factor
        doc_factor = min(len(retrieval_result.documents) / self.config.top_k, 1.0)
        
        # Combined confidence
        confidence = (avg_score * 0.7 + doc_factor * 0.3)
        
        return min(max(confidence, 0.0), 1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'is_initialized': self.is_initialized,
            'vector_store_stats': self.vector_store.get_stats() if self.vector_store else None,
            'config': {
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'top_k': self.config.top_k,
                'similarity_threshold': self.config.similarity_threshold,
            }
        }


def create_rag_pipeline(
    config: Optional[RAGConfig] = None,
    use_sample_kb: bool = True
) -> MedicalRAGPipeline:
    """
    Factory function to create a RAG pipeline.
    
    Args:
        config: RAG configuration
        use_sample_kb: Whether to load sample knowledge base
        
    Returns:
        Initialized MedicalRAGPipeline
    """
    pipeline = MedicalRAGPipeline(config=config)
    
    if use_sample_kb:
        pipeline.initialize()
    
    return pipeline


if __name__ == "__main__":
    # Test the RAG pipeline
    print("Initializing Medical RAG Pipeline...")
    
    pipeline = create_rag_pipeline()
    
    # Test query
    test_queries = [
        "What are the symptoms of hypertension?",
        "How is diabetes diagnosed?",
        "What is the treatment for myocardial infarction?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = pipeline.query(query)
        
        print(f"\nAnswer:\n{result.answer[:500]}...")
        print(f"\nConfidence: {result.confidence:.2f}")
        print(f"Total time: {result.total_time:.2f}s")
        print(f"Sources: {len(result.sources)}")
