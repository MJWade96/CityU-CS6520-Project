"""Native FAISS-backed vector store helpers used by the primary RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core import Document as LlamaDocument
from llama_index.core.ingestion.pipeline import run_transformations
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from ..data.json_utils import load_json_safe, save_json_atomic


@dataclass
class RetrievedDocument:
    """Lightweight document view used by the shared evaluation helpers."""

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchFaissVectorStore(FaissVectorStore):
    """Batch FAISS additions so embedded vectors reach FAISS in one array per batch."""

    def add(self, nodes: List[Any], **add_kwargs: Any) -> List[str]:
        del add_kwargs
        if not nodes:
            return []

        start_id = int(self.client.ntotal)
        embeddings = np.asarray(
            [node.get_embedding() for node in nodes],
            dtype="float32",
        )
        self.client.add(embeddings)
        return [str(start_id + offset) for offset in range(len(nodes))]


class MedicalVectorStore:
    """FAISS-backed native store with retriever and query-engine helpers."""

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-m3",
        normalize_embeddings: bool = True,
        batch_size: int = 256,
        use_gpu_faiss: bool = False,
        embedding_api_base_url: str = "",
        embedding_api_key: str = "",
        embedding_api_dimensions: Optional[int] = None,
        embedding_api_timeout: float = 120.0,
        embedding_api_max_retries: int = 5,
    ):
        self.embedding_backend = "api"
        self.embedding_model_name = embedding_model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.use_gpu_faiss = use_gpu_faiss
        self.embedding_api_base_url = embedding_api_base_url
        self.embedding_api_dimensions = embedding_api_dimensions
        self.embedding_api_timeout = embedding_api_timeout
        self.embedding_api_max_retries = embedding_api_max_retries
        self.index: Optional[VectorStoreIndex] = None
        if not embedding_api_base_url:
            raise ValueError("RAG_EMBEDDING_API_BASE_URL must be set for API embeddings")
        if not embedding_api_key:
            raise ValueError("RAG_EMBEDDING_API_KEY must be set for API embeddings")

        self._embed_model = OpenAIEmbedding(
            embed_batch_size=batch_size,
            model_name=embedding_model_name,
            api_base=embedding_api_base_url,
            api_key=embedding_api_key,
            dimensions=embedding_api_dimensions,
            timeout=embedding_api_timeout,
            max_retries=embedding_api_max_retries,
        )

    def _require_index(self) -> VectorStoreIndex:
        if self.index is None:
            raise ValueError("MedicalVectorStore has not been built or loaded")
        return self.index

    def _to_gpu_index(self, faiss_index: Any) -> Any:
        """Move FAISS storage to GPU only when the explicit GPU mode is available."""
        if not self.use_gpu_faiss:
            return faiss_index
        if not all(
            hasattr(faiss, attr)
            for attr in ("StandardGpuResources", "index_cpu_to_gpu", "get_num_gpus")
        ):
            raise RuntimeError(
                "GPU FAISS was requested, but the installed faiss package does not "
                "provide GPU APIs. Install a GPU-enabled FAISS build on AutoDL."
            )
        if faiss.get_num_gpus() < 1:
            raise RuntimeError("GPU FAISS was requested, but FAISS reports 0 GPUs.")
        resources = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(resources, 0, faiss_index)

    def _to_cpu_index_for_persist(self, faiss_index: Any) -> Any:
        """Convert GPU FAISS indexes back to CPU because FAISS persists CPU indexes."""
        if not self.use_gpu_faiss:
            return faiss_index
        if not hasattr(faiss, "index_gpu_to_cpu"):
            raise RuntimeError(
                "GPU FAISS persistence requires faiss.index_gpu_to_cpu, but it is missing."
            )
        return faiss.index_gpu_to_cpu(faiss_index)

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
        faiss_index = self._to_gpu_index(faiss.IndexFlatIP(dimension))
        vector_store = BatchFaissVectorStore(faiss_index=faiss_index)
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

        nodes = run_transformations(
            documents,
            self._require_index()._transformations,
            show_progress=show_progress,
        )
        self._require_index().insert_nodes(nodes, show_progress=show_progress)
        for document in documents:
            self._require_index().docstore.set_document_hash(
                document.id_,
                document.hash,
            )

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
        vector_store = self.index.storage_context.vector_store
        original_faiss_index = vector_store.client
        vector_store._faiss_index = self._to_cpu_index_for_persist(original_faiss_index)
        try:
            self.index.storage_context.persist(persist_dir=str(persist_dir))
        finally:
            vector_store._faiss_index = original_faiss_index
        save_json_atomic(
            persist_dir / "metadata.json",
            {
                "store_type": "native-faiss",
                "embedding_backend": self.embedding_backend,
                "embedding_model": self.embedding_model_name,
                "embedding_api_base_url": self.embedding_api_base_url,
                "embedding_api_dimensions": self.embedding_api_dimensions,
                "use_gpu_faiss": self.use_gpu_faiss,
            },
            indent=2,
            ensure_ascii=False,
        )

    def load(self, path: str) -> None:
        """Load a persisted native FAISS index."""
        persist_dir = Path(path)
        loaded_vector_store = FaissVectorStore.from_persist_dir(str(persist_dir))
        vector_store = BatchFaissVectorStore(
            faiss_index=self._to_gpu_index(loaded_vector_store.client)
        )
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
