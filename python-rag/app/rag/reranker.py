"""
Reranker Module

Implements document reranking strategies:
1. Cross-Encoder model-based reranking
"""

from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document

from .embeddings import resolve_torch_device


class CrossEncoderReranker:
    """
    Cross-Encoder based reranker
    
    Uses a bi-encoder model to score (query, document) pairs
    Recommended model: BAAI/bge-reranker-large
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "auto",
        top_k: int = 5,
    ):
        """
        Initialize Cross-Encoder reranker.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cpu' or 'cuda')
            top_k: Number of documents to return after reranking
        """
        self.model_name = model_name
        self.device = resolve_torch_device(device, env_var="RAG_RERANKER_DEVICE")
        self.top_k = top_k
        
        # Try to import sentence-transformers
        try:
            print(f"[Reranker] Loading Cross-Encoder model: {model_name} on {self.device}...")
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                model_name=model_name,
                device=self.device,
            )
            self.available = True
            print(f"Cross-Encoder loaded: {model_name}")
        except Exception as e:
            print(f"Cross-Encoder not available: {e}")
            self.model = None
            self.available = False
    
    def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using Cross-Encoder.
        
        Args:
            query: Query string
            documents: List of (document, initial_score) tuples
            top_k: Number of documents to return
            
        Returns:
            Reranked list of (document, new_score) tuples
        """
        if not self.available or not self.model:
            # Fallback: return original order
            return documents[:top_k or self.top_k]
        
        if not documents:
            return []
        
        # Prepare pairs for Cross-Encoder
        pairs = [
            [query, doc.page_content]
            for doc, _ in documents
        ]
        
        # Get relevance scores
        try:
            scores = self.model.predict(pairs)
            
            # Create reranked list
            reranked = [
                (doc, float(score))
                for (doc, _), score in zip(documents, scores)
            ]
            
            # Sort by score descending
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k
            k = top_k or self.top_k
            return reranked[:k]
            
        except Exception as e:
            print(f"Reranking failed: {e}")
            return documents[:top_k or self.top_k]


class RerankerPipeline:
    """
    Complete reranking pipeline

    Uses the active Cross-Encoder reranking path shared by evaluation scripts.
    """
    
    def __init__(
        self,
        use_cross_encoder: bool = True,
        cross_encoder_model: str = "BAAI/bge-reranker-large",
        cross_encoder_device: str = "auto",
        top_k: int = 5,
    ):
        """
        Initialize reranker pipeline.
        
        Args:
            use_cross_encoder: Use Cross-Encoder reranking
            cross_encoder_model: Cross-Encoder model name
            cross_encoder_device: Device for the Cross-Encoder model
            top_k: Final number of documents to return
        """
        self.top_k = top_k
        
        # Initialize rerankers
        if use_cross_encoder:
            self.cross_encoder = CrossEncoderReranker(
                model_name=cross_encoder_model,
                device=cross_encoder_device,
                top_k=top_k * 2  # Get more for final selection
            )
        else:
            self.cross_encoder = None
    
    def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using the pipeline.
        
        Args:
            query: Query string
            documents: List of (document, score) tuples
            
        Returns:
            Reranked list of (document, score) tuples
        """
        if not documents:
            return []
        
        current_docs = documents

        if self.cross_encoder and self.cross_encoder.available:
            reranked = self.cross_encoder.rerank(query, current_docs, top_k=self.top_k * 2)
            current_docs = reranked

        return current_docs[:self.top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'use_cross_encoder': self.cross_encoder is not None,
            'cross_encoder_available': self.cross_encoder.available if self.cross_encoder else False,
            'cross_encoder_model': self.cross_encoder.model_name if self.cross_encoder else None,
            'cross_encoder_device': self.cross_encoder.device if self.cross_encoder else None,
            'top_k': self.top_k,
        }
