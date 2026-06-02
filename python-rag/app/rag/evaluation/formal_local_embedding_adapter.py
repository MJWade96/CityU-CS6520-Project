"""Formal-only retrieval over precomputed local embedding artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from app.rag.data.corpus_registry import combine_registered_corpora
from app.rag.data.data_paths import RETRIEVAL_CACHE_DIR
from app.rag.data.json_utils import load_json_safe, save_json_atomic
from app.rag.evaluation.formal_artifacts import load_jsonl, write_json
from app.rag.retriever.vector_store import RetrievedDocument


FAISS_INDEX_FILENAME = "faiss.index"
FAISS_MANIFEST_FILENAME = "faiss_manifest.json"
BM25_CACHE_DIRNAME = "bm25_index"
BM25_MANIFEST_FILENAME = "manifest.json"
FORMAL_CACHE_CONFIG_VERSION = "formal_cache_design_v1"


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
    faiss_index: Any
    index_root: Path
    index_manifest: Dict[str, Any]
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
        faiss_index = _load_or_build_faiss_index(
            index_root=index_root,
            chunk_embeddings=chunk_embeddings,
            index_manifest=manifest,
        )
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
            faiss_index=faiss_index,
            index_root=index_root,
            index_manifest=manifest,
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
            self._bm25_by_k[k] = _load_or_build_bm25_retriever(
                index_root=self.index_root,
                nodes=self._as_nodes(),
                similarity_top_k=k,
                index_manifest=self.index_manifest,
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
        limit = min(max(1, k), int(self.faiss_index.ntotal))
        scores, indexes = self.faiss_index.search(query_vector.reshape(1, -1), limit)
        top_indexes = indexes[0]
        top_scores = scores[0]

        return [
            (self.documents[int(index)], float(score))
            for index, score in zip(top_indexes[:limit], top_scores[:limit])
            if int(index) >= 0
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


def _artifact_fingerprint(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_or_build_faiss_index(
    *,
    index_root: Path,
    chunk_embeddings: np.ndarray,
    index_manifest: Dict[str, Any],
) -> Any:
    index_path = index_root / FAISS_INDEX_FILENAME
    manifest_path = index_root / FAISS_MANIFEST_FILENAME
    chunk_path = index_root / "chunk_embeddings.npy"
    embedding_shape = [int(value) for value in chunk_embeddings.shape]

    if index_path.exists() and manifest_path.exists():
        manifest = load_json_safe(manifest_path)
        if (
            manifest.get("status") == "completed"
            and manifest.get("embedding_shape") == embedding_shape
            and manifest.get("faiss_index_type") == "FlatIP"
        ):
            return faiss.read_index(str(index_path))

    vectors = np.asarray(chunk_embeddings, dtype="float32")
    faiss_index = faiss.IndexFlatIP(int(vectors.shape[1]))
    faiss_index.add(vectors)
    faiss.write_index(faiss_index, str(index_path))
    save_json_atomic(
        manifest_path,
        {
            "artifact_id": f"{index_root.name}:faiss_flat",
            "artifact_group": "faiss_index",
            "status": "completed",
            "key": {
                "corpus_version": index_manifest.get("corpus_version"),
                "embedding_model": index_manifest.get("embedding_model"),
                "faiss_index_type": "FlatIP",
            },
            "input_artifacts": {
                "chunk_embeddings": str(chunk_path),
                "chunk_embedding_manifest": str(index_root / "manifest.json"),
            },
            "parameters": {"faiss_index_type": "FlatIP", "metric": "inner_product"},
            "embedding_shape": embedding_shape,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config_version": FORMAL_CACHE_CONFIG_VERSION,
            "dataset_split": "corpus",
            "fingerprint": {
                "chunk_embeddings": _artifact_fingerprint(chunk_path),
                "chunk_embedding_manifest": _artifact_fingerprint(index_root / "manifest.json"),
            },
        },
    )
    return faiss_index


def _load_or_build_bm25_retriever(
    *,
    index_root: Path,
    nodes: List[TextNode],
    similarity_top_k: int,
    index_manifest: Dict[str, Any],
) -> Any:
    bm25_cls = _import_bm25_retriever()
    cache_dir = index_root / BM25_CACHE_DIRNAME
    manifest_path = cache_dir / BM25_MANIFEST_FILENAME
    if cache_dir.exists() and manifest_path.exists():
        manifest = load_json_safe(manifest_path)
        if (
            manifest.get("status") == "completed"
            and manifest.get("document_count") == len(nodes)
        ):
            retriever = bm25_cls.from_persist_dir(str(cache_dir))
            retriever.similarity_top_k = similarity_top_k
            return retriever

    cache_dir.mkdir(parents=True, exist_ok=True)
    retriever = bm25_cls.from_defaults(nodes=nodes, similarity_top_k=similarity_top_k)
    retriever.persist(str(cache_dir))
    write_json(
        manifest_path,
        {
            "artifact_id": f"{index_root.name}:bm25",
            "artifact_group": "bm25_index",
            "status": "completed",
            "key": {
                "corpus_version": index_manifest.get("corpus_version"),
                "embedding_model": index_manifest.get("embedding_model"),
            },
            "input_artifacts": {
                "chunk_embedding_manifest": str(index_root / "manifest.json"),
            },
            "parameters": {"language": "en", "token_pattern": r"(?u)\b\w\w+\b"},
            "document_count": len(nodes),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config_version": FORMAL_CACHE_CONFIG_VERSION,
            "dataset_split": "corpus",
            "fingerprint": {
                "chunk_embedding_manifest": _artifact_fingerprint(index_root / "manifest.json"),
            },
        },
    )
    return retriever
