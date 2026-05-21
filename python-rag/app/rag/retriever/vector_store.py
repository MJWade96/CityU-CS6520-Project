"""Native FAISS-backed vector store helpers used by the primary RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core import Document as LlamaDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from tqdm import tqdm

from ..data.json_utils import load_json_safe, save_json_atomic


@dataclass
class RetrievedDocument:
    """Lightweight document view used by the shared evaluation helpers."""

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MedicalVectorStore:
    """FAISS-backed native store with retriever and query-engine helpers."""

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-m3",
        embedding_device: str = "cpu",
        normalize_embeddings: bool = True,
        batch_size: int = 256,
    ):
        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.index: Optional[VectorStoreIndex] = None
        self._embed_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            device=embedding_device,
            normalize=normalize_embeddings,
            embed_batch_size=batch_size,
        )

    def _require_index(self) -> VectorStoreIndex:
        if self.index is None:
            raise ValueError("MedicalVectorStore has not been built or loaded")
        return self.index

    def build(
        self,
        documents: List[LlamaDocument],
        *,
        show_progress: bool = False,
        insert_batch_size: int = 1024,
    ) -> None:
        """Build the native index from LlamaIndex documents."""
        if not documents:
            self.index = None
            return

        dimension = len(self._embed_model.get_text_embedding("dimension probe"))
        faiss_index = faiss.IndexFlatIP(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self._embed_model,
            show_progress=show_progress,
            insert_batch_size=insert_batch_size,
        )

    def add_documents(
        self,
        documents: List[LlamaDocument],
        *,
        show_progress: bool = False,
        insert_batch_size: int = 1024,
    ) -> None:
        """Add documents through the same native index path used for fresh builds."""
        if not documents:
            return

        if self.index is None:
            self.build(
                documents,
                show_progress=show_progress,
                insert_batch_size=insert_batch_size,
            )
            return

        document_iterator = (
            tqdm(
                documents,
                desc="Inserting documents",
                unit="doc",
                leave=False,
            )
            if show_progress
            else documents
        )
        for document in document_iterator:
            self.index.insert(document)

    def as_retriever(self, similarity_top_k: int = 5) -> Any:
        """Create a native retriever for the loaded index."""
        return self._require_index().as_retriever(similarity_top_k=similarity_top_k)

    def as_query_engine(self, *, llm: Any, similarity_top_k: int = 5) -> Any:
        """Create a native query engine for the loaded index."""
        return self._require_index().as_query_engine(
            llm=llm,
            similarity_top_k=similarity_top_k,
        )

    def retrieve(self, query: str, k: int = 5) -> List[Any]:
        """Expose native retrieval results for recall checks and smoke tests."""
        return self.as_retriever(similarity_top_k=k).retrieve(query)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[RetrievedDocument, float]]:
        """Return doc-and-score tuples compatible with shared evaluation helpers."""
        del filter

        results: List[Tuple[RetrievedDocument, float]] = []
        for node_with_score in self.retrieve(query, k=k):
            node = node_with_score.node
            results.append(
                (
                    RetrievedDocument(
                        page_content=node.get_content(),
                        metadata=dict(node.metadata),
                    ),
                    float(node_with_score.score or 0.0),
                )
            )
        return results

    def save(self, path: str) -> None:
        """Persist the index and lightweight metadata."""
        if self.index is None:
            return

        persist_dir = Path(path)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.index.storage_context.persist(persist_dir=str(persist_dir))
        save_json_atomic(
            persist_dir / "metadata.json",
            {
                "store_type": "native-faiss",
                "embedding_model": self.embedding_model_name,
                "embedding_device": self.embedding_device,
            },
            indent=2,
            ensure_ascii=False,
        )

    def load(self, path: str) -> None:
        """Load a persisted native FAISS index."""
        persist_dir = Path(path)
        vector_store = FaissVectorStore.from_persist_dir(str(persist_dir))
        storage_context = StorageContext.from_defaults(
            persist_dir=str(persist_dir),
            vector_store=vector_store,
        )
        self.index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=self._embed_model,
        )

        metadata_path = persist_dir / "metadata.json"
        if metadata_path.exists():
            metadata = load_json_safe(metadata_path)
            print(
                "Loaded vector store "
                f"with model {metadata.get('embedding_model', 'unknown')}"
            )
