"""
Reranker Module

Implements document reranking strategies:
1. Rule-based reranking (MMR, Diversity)
2. Cross-Encoder model-based reranking
3. LLM-based reranking (optional, high cost)
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document


class CrossEncoderReranker:
    """
    Cross-Encoder based reranker
    
    Uses a bi-encoder model to score (query, document) pairs
    Recommended model: BAAI/bge-reranker-large
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "cpu",
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
        self.device = device
        self.top_k = top_k
        
        # Try to import sentence-transformers
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                model_name=model_name,
                device=device,
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


class MMReranker:
    """
    Maximal Marginal Relevance (MMR) reranker
    
    Balances relevance and diversity
    """
    
    def __init__(self, lambda_mult: float = 0.5):
        """
        Initialize MMR reranker.
        
        Args:
            lambda_mult: Balance between relevance and diversity
                        1.0 = pure relevance
                        0.0 = pure diversity
        """
        self.lambda_mult = lambda_mult
    
    def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_k: int = 5,
        fetch_k: int = 20
    ) -> List[Document]:
        """
        Rerank using MMR.
        
        Args:
            query: Query string
            documents: List of (document, score) tuples
            top_k: Number of documents to return
            fetch_k: Number of candidates to consider
            
        Returns:
            List of diverse relevant documents
        """
        if not documents:
            return []
        
        # Use LangChain's built-in MMR if available
        # Otherwise, implement simple MMR
        selected = []
        remaining = list(documents[:fetch_k])
        
        while len(selected) < top_k and remaining:
            # Select document with highest MMR score
            best_idx = 0
            best_score = float('-inf')
            
            for i, (doc, score) in enumerate(remaining):
                # Calculate MMR score
                mmr_score = self._mmr_score(
                    query, doc, selected,
                    score, self.lambda_mult
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            # Add best document to selected
            selected.append(remaining.pop(best_idx)[0])
        
        return selected
    
    def _mmr_score(
        self,
        query: str,
        doc: Document,
        selected: List[Document],
        relevance_score: float,
        lambda_mult: float
    ) -> float:
        """
        Calculate MMR score for a document.
        
        MMR = λ * relevance - (1-λ) * max_similarity_to_selected
        """
        if not selected:
            return relevance_score
        
        # Calculate max similarity to already selected documents
        max_similarity = 0.0
        doc_text = doc.page_content.lower()
        
        for sel_doc in selected:
            sel_text = sel_doc.page_content.lower()
            similarity = self._text_similarity(doc_text, sel_text)
            max_similarity = max(max_similarity, similarity)
        
        # MMR score
        mmr_score = lambda_mult * relevance_score - (1 - lambda_mult) * max_similarity
        return mmr_score
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using Jaccard index"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


class LostInTheMiddleReranker:
    """
    LostInTheMiddle reranker
    
    Reorders documents to place most relevant at beginning and end
    to mitigate "lost in the middle" phenomenon in LLMs
    """
    
    def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_k: int = 5
    ) -> List[Document]:
        """
        Rerank using LostInTheMiddle strategy.
        
        Args:
            query: Query string
            documents: List of (document, score) tuples
            top_k: Number of documents to return
            
        Returns:
            Reordered list of documents
        """
        if not documents:
            return []
        
        # Sort by score
        sorted_docs = sorted(documents, key=lambda x: x[1], reverse=True)
        sorted_docs = sorted_docs[:top_k]
        
        # Reorder: most relevant first, then alternate end
        result = []
        left = 0
        right = len(sorted_docs) - 1
        
        while left <= right:
            if left == right:
                result.append(sorted_docs[left][0])
            else:
                result.append(sorted_docs[left][0])
                result.append(sorted_docs[right][0])
            left += 1
            right -= 1
        
        return result


class RerankerPipeline:
    """
    Complete reranking pipeline
    
    Combines multiple reranking strategies
    """
    
    def __init__(
        self,
        use_cross_encoder: bool = True,
        use_mmr: bool = False,
        use_lost_in_middle: bool = False,
        cross_encoder_model: str = "BAAI/bge-reranker-large",
        mmr_lambda: float = 0.5,
        top_k: int = 5,
    ):
        """
        Initialize reranker pipeline.
        
        Args:
            use_cross_encoder: Use Cross-Encoder reranking
            use_mmr: Use MMR for diversity
            use_lost_in_middle: Use LostInTheMiddle ordering
            cross_encoder_model: Cross-Encoder model name
            mmr_lambda: MMR lambda parameter
            top_k: Final number of documents to return
        """
        self.top_k = top_k
        
        # Initialize rerankers
        if use_cross_encoder:
            self.cross_encoder = CrossEncoderReranker(
                model_name=cross_encoder_model,
                top_k=top_k * 2  # Get more for final selection
            )
        else:
            self.cross_encoder = None
        
        if use_mmr:
            self.mmr = MMReranker(lambda_mult=mmr_lambda)
        else:
            self.mmr = None
        
        if use_lost_in_middle:
            self.lost_in_middle = LostInTheMiddleReranker()
        else:
            self.lost_in_middle = None
    
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
        
        # Step 1: Cross-Encoder reranking
        if self.cross_encoder and self.cross_encoder.available:
            reranked = self.cross_encoder.rerank(query, current_docs, top_k=self.top_k * 2)
            current_docs = reranked
        
        # Step 2: MMR for diversity (optional)
        if self.mmr:
            selected_docs = self.mmr.rerank(
                query,
                current_docs,
                top_k=self.top_k,
                fetch_k=min(len(current_docs), self.top_k * 2)
            )
            # Convert back to tuples with scores
            current_docs = [(doc, 0.0) for doc in selected_docs]
        
        # Step 3: LostInTheMiddle ordering (optional)
        if self.lost_in_middle:
            ordered_docs = self.lost_in_middle.rerank(
                query,
                current_docs,
                top_k=self.top_k
            )
            current_docs = [(doc, 0.0) for doc in ordered_docs]
        
        # Final: Return top-k
        return current_docs[:self.top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'use_cross_encoder': self.cross_encoder is not None,
            'cross_encoder_available': self.cross_encoder.available if self.cross_encoder else False,
            'use_mmr': self.mmr is not None,
            'use_lost_in_middle': self.lost_in_middle is not None,
            'top_k': self.top_k,
        }


if __name__ == "__main__":
    # Test rerankers
    print("Testing Reranker Module...")
    
    from langchain_core.documents import Document
    
    # Create test documents
    docs = [
        (Document(page_content="Hypertension is high blood pressure."), 0.9),
        (Document(page_content="Diabetes affects blood sugar."), 0.85),
        (Document(page_content="High blood pressure can cause headaches."), 0.8),
        (Document(page_content="Pneumonia is a lung infection."), 0.75),
        (Document(page_content="Blood pressure medication includes ACE inhibitors."), 0.7),
    ]
    
    query = "What is hypertension?"
    
    # Test Cross-Encoder
    print("\n=== Cross-Encoder Reranker ===")
    ce_reranker = CrossEncoderReranker(top_k=3)
    
    if ce_reranker.available:
        reranked = ce_reranker.rerank(query, docs)
        print(f"Reranked results:")
        for i, (doc, score) in enumerate(reranked, 1):
            print(f"  {i}. Score: {score:.4f} | {doc.page_content[:50]}...")
    else:
        print("Cross-Encoder not available, skipping test")
    
    # Test MMR
    print("\n=== MMR Reranker ===")
    mmr_reranker = MMReranker(lambda_mult=0.7)
    
    selected = mmr_reranker.rerank(query, docs, top_k=3)
    print(f"MMR selected:")
    for i, doc in enumerate(selected, 1):
        print(f"  {i}. {doc.page_content[:50]}...")
    
    # Test LostInTheMiddle
    print("\n=== LostInTheMiddle Reranker ===")
    litm_reranker = LostInTheMiddleReranker()
    
    ordered = litm_reranker.rerank(query, docs, top_k=5)
    print(f"Ordered results:")
    for i, doc in enumerate(ordered, 1):
        print(f"  {i}. {doc.page_content[:50]}...")
    
    # Test pipeline
    print("\n=== Complete Reranker Pipeline ===")
    pipeline = RerankerPipeline(
        use_cross_encoder=True,
        use_mmr=False,
        use_lost_in_middle=False,
        top_k=3
    )
    
    stats = pipeline.get_stats()
    print(f"Pipeline config: {stats}")
    
    if stats['cross_encoder_available']:
        reranked = pipeline.rerank(query, docs)
        print(f"\nFinal reranked results:")
        for i, (doc, score) in enumerate(reranked, 1):
            print(f"  {i}. Score: {score:.4f} | {doc.page_content[:50]}...")
