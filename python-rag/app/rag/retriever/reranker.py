"""Native reranker helpers for the enhanced RAG path."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, TextNode

from .runtime_config import resolve_torch_device
from .vector_store import RetrievedDocument


class CrossEncoderReranker:
    """SentenceTransformer-backed native reranker."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "auto",
        top_k: int = 5,
    ):
        self.model_name = model_name
        self.device = resolve_torch_device(device, env_var="RAG_RERANKER_DEVICE")
        self.top_k = top_k

        try:
            self.model = SentenceTransformerRerank(
                top_n=top_k,
                model=model_name,
                device=self.device,
            )
            self.available = True
        except Exception as exc:
            print(f"SentenceTransformerRerank not available: {exc}")
            self.model = None
            self.available = False

    def _to_nodes(
        self,
        documents: List[Tuple[RetrievedDocument, float]],
    ) -> List[NodeWithScore]:
        nodes: List[NodeWithScore] = []
        for document, score in documents:
            nodes.append(
                NodeWithScore(
                    node=TextNode(
                        text=document.page_content,
                        extra_info=dict(document.metadata),
                    ),
                    score=float(score),
                )
            )
        return nodes

    def rerank(
        self,
        query: str,
        documents: List[Tuple[RetrievedDocument, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[RetrievedDocument, float]]:
        if not self.available or self.model is None:
            return documents[: top_k or self.top_k]

        reranked = self.model.postprocess_nodes(
            self._to_nodes(documents),
            query_str=query,
        )
        limit = top_k or self.top_k
        return [
            (
                RetrievedDocument(
                    page_content=node_with_score.node.get_content(),
                    metadata=dict(node_with_score.node.metadata),
                ),
                float(node_with_score.score or 0.0),
            )
            for node_with_score in reranked[:limit]
        ]


class RerankerPipeline:
    """Reranker pipeline that preserves the old public surface."""

    def __init__(
        self,
        use_cross_encoder: bool = True,
        cross_encoder_model: str = "BAAI/bge-reranker-large",
        cross_encoder_device: str = "auto",
        top_k: int = 5,
    ):
        self.top_k = top_k
        self.cross_encoder = (
            CrossEncoderReranker(
                model_name=cross_encoder_model,
                device=cross_encoder_device,
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
            "cross_encoder_available": (
                self.cross_encoder.available if self.cross_encoder is not None else False
            ),
            "cross_encoder_model": (
                self.cross_encoder.model_name if self.cross_encoder is not None else None
            ),
            "cross_encoder_device": (
                self.cross_encoder.device if self.cross_encoder is not None else None
            ),
            "top_k": self.top_k,
        }