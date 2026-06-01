"""Formal-only retrieval over precomputed local embedding artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from app.rag.data.corpus_registry import combine_registered_corpora
from app.rag.data.data_paths import RETRIEVAL_CACHE_DIR
from app.rag.data.json_utils import load_json_safe
from app.rag.evaluation.formal_artifacts import load_jsonl
from app.rag.retriever.vector_store import RetrievedDocument


def _import_bm25_retriever() -> Any:
    try:
        from llama_index.retrievers.bm25 import BM25Retriever
    except ImportError as exc:
        raise ImportError(
            "Formal local embedding hybrid retrieval requires llama-index-retrievers-bm25."
        ) from exc
    return BM25Retriever


class _LocalDenseRetriever(BaseRetriever):
    """LlamaIndex retriever wrapper over precomputed query/chunk embeddings."""

    def __init__(
        self,
        adapter: "LocalEmbeddingFormalRetriever",
        *,
        question_id: str,
        similarity_top_k: int,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.question_id = question_id
        self.similarity_top_k = similarity_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        results = self.adapter.retrieve(
            question_id=self.question_id,
            query_text=query_bundle.query_str,
            k=self.similarity_top_k,
        )
        return [
            NodeWithScore(
                node=TextNode(text=document.page_content, metadata=dict(document.metadata)),
                score=score,
            )
            for document, score in results
        ]


@dataclass
class LocalEmbeddingFormalRetriever:
    """Retrieve from AutoDL-generated embedding arrays without touching the main store."""

    documents: List[RetrievedDocument]
    chunk_embeddings: np.ndarray
    query_embeddings: np.ndarray
    query_row_by_id: Dict[str, int]
    query_text_by_id: Dict[str, str]
    _nodes: List[TextNode] | None = field(default=None, init=False, repr=False)
    _bm25_by_k: Dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def load(
        cls,
        *,
        corpus_version: str,
        index_root: Path,
        query_cache_id: str,
    ) -> "LocalEmbeddingFormalRetriever":
        chunk_path = index_root / "chunk_embeddings.npy"
        index_manifest_path = index_root / "manifest.json"
        query_dir = RETRIEVAL_CACHE_DIR / query_cache_id
        query_path = query_dir / "query_embeddings.npy"
        query_texts_path = query_dir / "query_texts.jsonl"

        for required_path in (
            chunk_path,
            index_manifest_path,
            query_path,
            query_texts_path,
        ):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Required formal local embedding artifact is missing: {required_path}"
                )

        manifest = load_json_safe(index_manifest_path)
        selected_sources = manifest.get("selected_sources")
        if not selected_sources:
            raise ValueError(f"Local embedding manifest lacks selected_sources: {index_manifest_path}")

        corpus = combine_registered_corpora(selected_sources=selected_sources)
        documents: List[RetrievedDocument] = []
        for record in corpus["records"]:
            content = str(record.get("contents") or record.get("content") or "").strip()
            if not content:
                continue
            documents.append(
                RetrievedDocument(
                    page_content=content,
                    metadata={
                        "doc_id": record.get("id"),
                        "title": record.get("title"),
                        "source": record.get("source"),
                        "corpus_version": corpus_version,
                    },
                )
            )
        chunk_embeddings = np.load(chunk_path, mmap_mode="r")
        query_embeddings = np.load(query_path, mmap_mode="r")
        if len(documents) != int(chunk_embeddings.shape[0]):
            raise ValueError(
                f"Document count {len(documents)} does not match embedding rows "
                f"{chunk_embeddings.shape[0]} for {index_root}"
            )

        query_rows = load_jsonl(query_texts_path)
        query_row_by_id = {
            str(row["question_id"]): index
            for index, row in enumerate(query_rows)
            if row.get("question_id") is not None
        }
        query_text_by_id = {
            str(row["question_id"]): str(row.get("query_text") or "")
            for row in query_rows
            if row.get("question_id") is not None
        }
        if len(query_row_by_id) != int(query_embeddings.shape[0]):
            raise ValueError(
                f"Query id count {len(query_row_by_id)} does not match embedding rows "
                f"{query_embeddings.shape[0]} for {query_dir}"
            )

        return cls(
            documents=documents,
            chunk_embeddings=chunk_embeddings,
            query_embeddings=query_embeddings,
            query_row_by_id=query_row_by_id,
            query_text_by_id=query_text_by_id,
        )

    def cached_query_text(self, question_id: str) -> str:
        if question_id not in self.query_text_by_id:
            raise KeyError(f"No local embedding query text for question_id={question_id}")
        return self.query_text_by_id[question_id]

    def _as_nodes(self) -> List[TextNode]:
        if self._nodes is None:
            self._nodes = [
                TextNode(text=document.page_content, metadata=dict(document.metadata))
                for document in self.documents
            ]
        return self._nodes

    def _bm25_retriever(self, k: int) -> Any:
        if k not in self._bm25_by_k:
            self._bm25_by_k[k] = _import_bm25_retriever().from_defaults(
                nodes=self._as_nodes(),
                similarity_top_k=k,
            )
        return self._bm25_by_k[k]

    def retrieve(
        self,
        *,
        question_id: str,
        query_text: str,
        k: int,
    ) -> List[Tuple[RetrievedDocument, float]]:
        del query_text
        if question_id not in self.query_row_by_id:
            raise KeyError(f"No local embedding query vector for question_id={question_id}")

        query_vector = np.asarray(
            self.query_embeddings[self.query_row_by_id[question_id]],
            dtype="float32",
        )
        scores = np.asarray(self.chunk_embeddings @ query_vector, dtype="float32")
        limit = min(max(1, k), len(scores))
        if limit == len(scores):
            top_indexes = np.argsort(-scores)
        else:
            candidates = np.argpartition(-scores, limit - 1)[:limit]
            top_indexes = candidates[np.argsort(-scores[candidates])]

        return [
            (self.documents[int(index)], float(scores[int(index)]))
            for index in top_indexes[:limit]
        ]

    def retrieve_components(
        self,
        *,
        question_id: str,
        query_text: str,
        k: int,
        weights: Tuple[float, float],
        llm: Any,
    ) -> Tuple[List[NodeWithScore], List[NodeWithScore], List[NodeWithScore]]:
        """Return dense, BM25, and QueryFusion results for Advanced formal rows."""
        if llm is None:
            raise ValueError("Formal local embedding QueryFusion requires an explicit LlamaIndex LLM")
        dense_retriever = _LocalDenseRetriever(
            self,
            question_id=question_id,
            similarity_top_k=k,
        )
        sparse_retriever = self._bm25_retriever(k)
        fusion_retriever = QueryFusionRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            llm=llm,
            similarity_top_k=k,
            num_queries=1,
            use_async=False,
            retriever_weights=list(weights),
        )
        return (
            dense_retriever.retrieve(query_text),
            list(sparse_retriever.retrieve(query_text)),
            list(fusion_retriever.retrieve(query_text)),
        )
