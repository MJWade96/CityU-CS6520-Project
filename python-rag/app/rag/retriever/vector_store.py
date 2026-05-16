"""
Vector Store Module
Handles document storage and retrieval using the FAISS vector store
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from ..data.json_utils import load_json_safe, save_json_atomic


class MedicalVectorStore:
    """
    Medical Vector Store using LangChain FAISS.

    The current project only builds and evaluates against the persisted FAISS
    index, so the wrapper stays focused on that one backend.
    """
    
    def __init__(
        self,
        embedding_model: Embeddings,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: LangChain embeddings instance
            distance_strategy: Distance metric for similarity search
        """
        self.embedding_model = embedding_model
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
            self.vectorstore = FAISS.from_documents(
                documents,
                self.embedding_model,
                distance_strategy=self.distance_strategy,
            )
        else:
            self.vectorstore.add_documents(documents)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores.

        Keeping this logic in one method avoids duplicating FAISS search calls
        across evaluation scripts.
        """
        if self.vectorstore is None:
            return []

        return self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter
        )
    
    def save(self, path: str) -> None:
        """Save the vector store to disk"""
        if self.vectorstore is None:
            return
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(str(path))
        
        # Save metadata
        metadata = {
            'store_type': 'faiss',
            'document_count': len(self.documents),
        }
        save_json_atomic(path / "metadata.json", metadata, indent=2, ensure_ascii=False)
    
    def load(self, path: str) -> None:
        """Load the vector store from disk"""
        path = Path(path)

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
            metadata = load_json_safe(metadata_path)
            print(f"Loaded vector store with {metadata.get('document_count', 0)} documents")
