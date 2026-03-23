"""
Hybrid Retriever Module

Implements hybrid retrieval combining:
1. Dense retrieval (semantic similarity)
2. Sparse retrieval (BM25 keyword matching)
3. RRF (Reciprocal Rank Fusion) for merging results
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import math

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse retrieval
    
    Supports:
    - Dense retrieval using embeddings
    - BM25 sparse retrieval
    - RRF fusion for combining results
    """
    
    def __init__(
        self,
        embedding_model: Embeddings,
        documents: List[Document],
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        dense_weight: float = 0.5,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            embedding_model: LangChain embeddings instance
            documents: List of documents to index
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            dense_weight: Weight for dense retrieval (1-dense_weight for sparse)
        """
        self.embedding_model = embedding_model
        self.documents = documents
        self.dense_weight = dense_weight
        
        # Initialize BM25
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        
        # Tokenize documents for BM25
        self.bm25_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(
            self.bm25_corpus,
            k1=bm25_k1,
            b=bm25_b
        )
        
        # Build dense index
        self._build_dense_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25"""
        # Lowercase and split on whitespace and punctuation
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _build_dense_index(self):
        """Build dense embeddings for all documents"""
        texts = [doc.page_content for doc in self.documents]
        self.dense_embeddings = self.embedding_model.embed_documents(texts)
    
    def _dense_search(
        self,
        query: str,
        k: int
    ) -> List[Tuple[int, float]]:
        """
        Perform dense retrieval.
        
        Returns:
            List of (doc_index, score) tuples
        """
        query_embedding = self.embedding_model.embed_query(query)
        
        # Calculate cosine similarity
        scores = []
        for i, doc_emb in enumerate(self.dense_embeddings):
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(query_embedding, doc_emb))
            norm_query = math.sqrt(sum(x * x for x in query_embedding))
            norm_doc = math.sqrt(sum(x * x for x in doc_emb))
            
            if norm_query > 0 and norm_doc > 0:
                similarity = dot_product / (norm_query * norm_doc)
            else:
                similarity = 0.0
            
            scores.append((i, similarity))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def _sparse_search(
        self,
        query: str,
        k: int
    ) -> List[Tuple[int, float]]:
        """
        Perform sparse (BM25) retrieval.
        
        Returns:
            List of (doc_index, score) tuples
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Convert to list of (index, score)
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        
        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:k]
    
    def _rrf_fusion(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        k: int,
        rrf_k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion (RRF) for merging results.
        
        RRF Score = Σ 1 / (k + rank_i)
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            k: Final number of results
            rrf_k: RRF constant (typically 60)
            
        Returns:
            Fused results as (doc_index, rrf_score)
        """
        # Create rank mappings
        dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(dense_results)}
        sparse_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sparse_results)}
        
        # Get all document IDs
        all_doc_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        for doc_id in all_doc_ids:
            score = 0.0
            
            # Dense contribution
            if doc_id in dense_ranks:
                score += self.dense_weight * (1.0 / (rrf_k + dense_ranks[doc_id]))
            
            # Sparse contribution
            if doc_id in sparse_ranks:
                score += (1 - self.dense_weight) * (1.0 / (rrf_k + sparse_ranks[doc_id]))
            
            rrf_scores[doc_id] = score
        
        # Sort by RRF score
        fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return fused_results[:k]
    
    def search(
        self,
        query: str,
        k: int = 5,
        use_hybrid: bool = True,
        rrf_k: int = 60
    ) -> List[Tuple[Document, float]]:
        """
        Search using hybrid retrieval.
        
        Args:
            query: Query string
            k: Number of results to return
            use_hybrid: Whether to use hybrid retrieval (True) or just dense (False)
            rrf_k: RRF constant
            
        Returns:
            List of (document, score) tuples
        """
        if not use_hybrid:
            # Dense only
            dense_results = self._dense_search(query, k * 2)
            results = [(self.documents[doc_id], score) for doc_id, score in dense_results[:k]]
            return results
        
        # Hybrid retrieval
        # Fetch more candidates for fusion
        fetch_k = k * 3
        
        dense_results = self._dense_search(query, fetch_k)
        sparse_results = self._sparse_search(query, fetch_k)
        
        # Fuse results
        fused_results = self._rrf_fusion(dense_results, sparse_results, k, rrf_k)
        
        # Convert to Document objects
        final_results = [
            (self.documents[doc_id], score)
            for doc_id, score in fused_results
        ]
        
        return final_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            'total_documents': len(self.documents),
            'dense_embedding_dim': len(self.dense_embeddings[0]) if self.dense_embeddings else 0,
            'bm25_params': {
                'k1': self.bm25_k1,
                'b': self.bm25_b
            },
            'dense_weight': self.dense_weight
        }


class AdaptiveRetriever:
    """
    Adaptive retriever that decides whether to retrieve based on query type
    """
    
    def __init__(self, hybrid_retriever: HybridRetriever):
        """Initialize adaptive retriever"""
        self.hybrid_retriever = hybrid_retriever
        self.non_retrieval_queries = {
            '你好', '您好', '谢谢', '感谢', '再见',
            'hello', 'hi', 'thanks', 'bye', 'thank you'
        }
    
    def should_retrieve(self, query: str) -> bool:
        """
        Decide if retrieval is needed for this query.
        
        Uses rule-based filtering:
        - Skip greetings and simple acknowledgments
        - Retrieve for medical questions
        """
        query_lower = query.strip().lower()
        
        # Check against non-retrieval queries
        if query_lower in self.non_retrieval_queries:
            return False
        
        # Check for question patterns
        question_words = ['什么', '如何', '怎样', '怎么', '为什么', '哪些', '是否', '吗', '?']
        if any(word in query_lower for word in question_words):
            return True
        
        # Check for medical terms (simplified)
        medical_indicators = [
            '症状', '治疗', '诊断', '疾病', '药物', '医学',
            '病', '药', '医', '疗法', '处方', '剂量'
        ]
        if any(term in query_lower for term in medical_indicators):
            return True
        
        # Default: retrieve for longer queries
        if len(query_lower) > 10:
            return True
        
        return False
    
    def search(
        self,
        query: str,
        k: int = 5,
        force_retrieve: bool = False
    ) -> List[Tuple[Document, float]]:
        """
        Search with adaptive retrieval.
        
        Args:
            query: Query string
            k: Number of results
            force_retrieve: Force retrieval even if judge says no
            
        Returns:
            List of (document, score) tuples
        """
        if not force_retrieve and not self.should_retrieve(query):
            return []
        
        return self.hybrid_retriever.search(query, k)


if __name__ == "__main__":
    # Test hybrid retriever
    from langchain_core.documents import Document
    from app.rag.embeddings import get_langchain_embeddings
    
    print("Testing Hybrid Retriever...")
    
    # Create test documents
    docs = [
        Document(page_content="Hypertension is high blood pressure. Symptoms include headache and dizziness."),
        Document(page_content="Diabetes affects blood sugar levels. Treatment includes insulin."),
        Document(page_content="Pneumonia is a lung infection. Symptoms include cough and fever."),
        Document(page_content="Heart disease can cause chest pain and shortness of breath."),
        Document(page_content="Cancer treatment includes chemotherapy and radiation therapy."),
    ]
    
    # Initialize embeddings
    embeddings = get_langchain_embeddings(model_type="huggingface")
    
    # Create hybrid retriever
    retriever = HybridRetriever(
        embedding_model=embeddings,
        documents=docs,
        dense_weight=0.5
    )
    
    # Test search
    query = "What is high blood pressure?"
    print(f"\nQuery: {query}")
    
    results = retriever.search(query, k=3, use_hybrid=True)
    print(f"\nHybrid Results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. Score: {score:.4f} | {doc.page_content[:60]}...")
    
    # Test dense-only
    results_dense = retriever.search(query, k=3, use_hybrid=False)
    print(f"\nDense-only Results:")
    for i, (doc, score) in enumerate(results_dense, 1):
        print(f"  {i}. Score: {score:.4f} | {doc.page_content[:60]}...")
    
    # Test adaptive retriever
    print(f"\nTesting Adaptive Retriever:")
    adaptive = AdaptiveRetriever(retriever)
    
    test_queries = [
        "你好",
        "What are symptoms of hypertension?",
        "谢谢",
        "How to treat diabetes?"
    ]
    
    for q in test_queries:
        should_retrieve = adaptive.should_retrieve(q)
        print(f"  Query: '{q}' -> Retrieve: {should_retrieve}")
