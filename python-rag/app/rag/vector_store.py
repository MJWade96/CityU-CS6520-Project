"""
Vector Store Module
Handles document storage and retrieval using LangChain vector stores
"""

import os
import json
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.vectorstores.utils import DistanceStrategy


class MedicalVectorStore:
    """
    Medical Vector Store using LangChain
    
    Provides efficient storage and retrieval of medical document embeddings
    using FAISS or Chroma as the backend.
    """
    
    def __init__(
        self,
        embedding_model: Embeddings,
        store_type: str = "faiss",
        persist_directory: Optional[str] = None,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: LangChain embeddings instance
            store_type: Type of vector store ('faiss' or 'chroma')
            persist_directory: Directory to persist the vector store
            distance_strategy: Distance metric for similarity search
        """
        self.embedding_model = embedding_model
        self.store_type = store_type
        self.persist_directory = persist_directory
        self.distance_strategy = distance_strategy
        
        self.vectorstore: Optional[Any] = None
        self.documents: List[Document] = []
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            ids: Optional list of document IDs
        """
        self.documents.extend(documents)
        
        if self.vectorstore is None:
            # Create new vector store
            if self.store_type == "faiss":
                self.vectorstore = FAISS.from_documents(
                    documents,
                    self.embedding_model,
                    distance_strategy=self.distance_strategy,
                )
            elif self.store_type == "chroma":
                self.vectorstore = Chroma.from_documents(
                    documents,
                    self.embedding_model,
                    persist_directory=self.persist_directory,
                )
            else:
                raise ValueError(f"Unknown store type: {self.store_type}")
        else:
            # Add to existing vector store
            if self.store_type == "faiss":
                self.vectorstore.add_documents(documents)
            elif self.store_type == "chroma":
                self.vectorstore.add_documents(documents)
        
        # Persist if directory is specified
        if self.persist_directory and self.store_type == "faiss":
            self.save(self.persist_directory)
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            return []
        
        return self.vectorstore.similarity_search(
            query,
            k=k,
            filter=filter
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores.

        Keeping this logic in one method avoids duplicate search behavior for
        FAISS and Chroma backends.
        """
        if self.vectorstore is None:
            return []

        if self.store_type == "faiss":
            return self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=filter
            )
        if self.store_type == "chroma":
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query,
                k=k,
                filter=filter
            )
            return [(doc, 1.0 - score) for doc, score in results]

        return []
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Perform MMR search for diverse results.
        
        Args:
            query: Query string
            k: Number of results to return
            fetch_k: Number of candidates to fetch
            lambda_mult: Balance between relevance and diversity
            
        Returns:
            List of diverse relevant documents
        """
        if self.vectorstore is None:
            return []
        
        return self.vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
    
    def save(self, path: str) -> None:
        """Save the vector store to disk"""
        if self.vectorstore is None:
            return
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.store_type == "faiss":
            self.vectorstore.save_local(str(path))
        
        # Save metadata
        metadata = {
            'store_type': self.store_type,
            'document_count': len(self.documents),
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    def load(self, path: str) -> None:
        """Load the vector store from disk"""
        path = Path(path)
        
        if self.store_type == "faiss":
            self.vectorstore = FAISS.load_local(
                str(path),
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            # Restore documents from docstore
            if hasattr(self.vectorstore, 'docstore') and self.vectorstore.docstore:
                self.documents = list(self.vectorstore.docstore._dict.values())
        
        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print(f"Loaded vector store with {metadata.get('document_count', 0)} documents")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'store_type': self.store_type,
            'document_count': len(self.documents),
            'is_initialized': self.vectorstore is not None,
        }
    
    def clear(self) -> None:
        """Clear all documents from the vector store"""
        self.vectorstore = None
        self.documents = []


class VectorStoreManager:
    """
    Manager for multiple vector stores
    
    Allows managing multiple vector stores for different document types
    or categories.
    """
    
    def __init__(self, embedding_model: Embeddings):
        """Initialize the manager"""
        self.embedding_model = embedding_model
        self.stores: Dict[str, MedicalVectorStore] = {}
    
    def get_store(
        self,
        name: str,
        store_type: str = "faiss"
    ) -> MedicalVectorStore:
        """Get or create a vector store by name"""
        if name not in self.stores:
            self.stores[name] = MedicalVectorStore(
                embedding_model=self.embedding_model,
                store_type=store_type
            )
        return self.stores[name]
    
    def search_all(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Search across all vector stores"""
        all_results = []
        
        for store in self.stores.values():
            results = store.similarity_search_with_score(query, k=k)
            all_results.extend(results)
        
        # Sort by score and return top k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]


if __name__ == "__main__":
    # Test the vector store
    from langchain_core.documents import Document
    from embeddings import get_langchain_embeddings

    # Create embeddings (use HuggingFace for real embeddings)
    embeddings = get_langchain_embeddings(model_type="huggingface")

    # Create vector store
    store = MedicalVectorStore(embeddings)
    
    # Add test documents
    docs = [
        Document(page_content="Hypertension is high blood pressure.", metadata={"category": "cardiovascular"}),
        Document(page_content="Diabetes affects blood sugar levels.", metadata={"category": "endocrine"}),
        Document(page_content="Pneumonia is a lung infection.", metadata={"category": "respiratory"}),
    ]
    
    store.add_documents(docs)
    
    # Test search
    results = store.similarity_search("blood pressure", k=2)
    print(f"Found {len(results)} documents")
    for doc in results:
        print(f"  - {doc.page_content[:50]}...")
