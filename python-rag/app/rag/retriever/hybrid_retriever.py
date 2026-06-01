"""Native hybrid retriever helpers for the enhanced RAG path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from llama_index.core.retrievers import QueryFusionRetriever

from .vector_store import MedicalVectorStore, RetrievedDocument


def _import_bm25_retriever() -> Any:
    try:
        from llama_index.retrievers.bm25 import BM25Retriever
    except ImportError as exc:
        raise ImportError(
            "Hybrid retrieval requires llama-index-retrievers-bm25."
        ) from exc
    return BM25Retriever


@dataclass(frozen=True)
class HybridRetrieverConfig:
    similarity_top_k: int = 5
    num_queries: int = 1
    retriever_weights: Optional[Tuple[float, float]] = None
    use_async: bool = False


class HybridRetriever:
    """Hybrid retriever built from native dense and BM25 retrievers."""

    def __init__(
        self,
        *,
        dense_retriever: Any,
        bm25_retriever: Any,
        fusion_retriever: Any,
        config: HybridRetrieverConfig,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.fusion_retriever = fusion_retriever
        self.config = config

    @classmethod
    def from_vector_store(
        cls,
        vectorstore: MedicalVectorStore,
        *,
        llm: Optional[Any] = None,
        similarity_top_k: int = 5,
        num_queries: int = 1,
        retriever_weights: Optional[Tuple[float, float]] = None,
        use_async: bool = False,
    ) -> "HybridRetriever":
        index = vectorstore._require_index()
        dense_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        bm25_retriever = _import_bm25_retriever().from_defaults(
            index=index,
            similarity_top_k=similarity_top_k,
        )
        fusion_retriever = QueryFusionRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            llm=llm,
            similarity_top_k=similarity_top_k,
            num_queries=max(1, num_queries),
            use_async=use_async,
            retriever_weights=(
                list(retriever_weights) if retriever_weights is not None else None
            ),
        )
        return cls(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            fusion_retriever=fusion_retriever,
            config=HybridRetrieverConfig(
                similarity_top_k=similarity_top_k,
                num_queries=max(1, num_queries),
                retriever_weights=retriever_weights,
                use_async=use_async,
            ),
        )

    def retrieve(self, query: str, *, use_hybrid: bool = True) -> List[Any]:
        retriever = self.fusion_retriever if use_hybrid else self.dense_retriever
        return list(retriever.retrieve(query))

    def retrieve_components(self, query: str) -> Tuple[List[Any], List[Any], List[Any]]:
        """Expose dense, BM25, and fused results for formal cache artifacts."""
        dense = list(self.dense_retriever.retrieve(query))
        sparse = list(self.bm25_retriever.retrieve(query))
        fusion = list(self.fusion_retriever.retrieve(query))
        return dense, sparse, fusion

    def search(
        self,
        query: str,
        k: int = 5,
        use_hybrid: bool = True,
        rrf_k: int = 60,
    ) -> List[Tuple[RetrievedDocument, float]]:
        del rrf_k

        results: List[Tuple[RetrievedDocument, float]] = []
        for node_with_score in self.retrieve(query, use_hybrid=use_hybrid)[:k]:
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
