"""API reranker helpers for the enhanced RAG path."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank

from .vector_store import RetrievedDocument


class SiliconFlowReranker:
    """Small wrapper matching the existing pipeline surface around LlamaIndex rerank."""

    def __init__(
        self,
        *,
        model_name: str,
        api_url: str,
        api_key: str,
        top_k: int,
    ) -> None:
        if not api_url:
            raise ValueError("RAG_RERANKER_API_URL must be set for API reranking")
        if not api_key:
            raise ValueError("RAG_RERANKER_API_KEY must be set for API reranking")

        self.model_name = model_name
        self.device = "api"
        self.top_k = top_k
        self.model = SiliconFlowRerank(
            model=model_name,
            base_url=api_url,
            api_key=api_key,
            top_n=top_k,
            return_documents=False,
        )
        self.available = True

    def _to_nodes(
        self,
        documents: List[Tuple[RetrievedDocument, float]],
    ) -> List[NodeWithScore]:
        return [
            NodeWithScore(
                node=TextNode(
                    text=document.page_content,
                    metadata=dict(document.metadata),
                ),
                score=float(score),
            )
            for document, score in documents
        ]

    def rerank(
        self,
        query: str,
        documents: List[Tuple[RetrievedDocument, float]],
        top_k: int | None = None,
    ) -> List[Tuple[RetrievedDocument, float]]:
        nodes = self.model.postprocess_nodes(self._to_nodes(documents), query_str=query)
        limit = top_k or self.top_k
        return [
            (
                RetrievedDocument(
                    page_content=node_with_score.node.get_content(),
                    metadata=dict(node_with_score.node.metadata),
                ),
                float(node_with_score.score or 0.0),
            )
            for node_with_score in nodes[:limit]
        ]


class RerankerPipeline:
    """Reranker pipeline backed only by the configured rerank API."""

    def __init__(
        self,
        use_cross_encoder: bool = True,
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        top_k: int = 5,
        api_url: str = "",
        api_key: str = "",
    ):
        self.top_k = top_k
        self.cross_encoder = (
            SiliconFlowReranker(
                model_name=cross_encoder_model,
                api_url=api_url,
                api_key=api_key,
                top_k=top_k,
            )
            if use_cross_encoder
            else None
        )

    def rerank(
        self,
        query: str,
        documents: List[Tuple[RetrievedDocument, float]],
    ) -> List[Tuple[RetrievedDocument, float]]:
        if not documents:
            return []

        if self.cross_encoder is None:
            return documents[: self.top_k]

        return self.cross_encoder.rerank(query, documents, top_k=self.top_k)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "use_cross_encoder": self.cross_encoder is not None,
            "cross_encoder_backend": "api",
            "cross_encoder_available": (
                self.cross_encoder.available if self.cross_encoder is not None else False
            ),
            "cross_encoder_model": (
                self.cross_encoder.model_name if self.cross_encoder is not None else None
            ),
            "top_k": self.top_k,
        }
