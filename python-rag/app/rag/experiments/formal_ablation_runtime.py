"""Execution helpers for formal phase-1 ablation artifacts.

This module keeps experiment-only embedding caches out of the primary RAG
pipeline. The main retriever stack remains API-only through LlamaIndex; MedCPT
is only used here to generate offline experiment artifacts.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import faiss
import numpy as np
from openai import OpenAI

from app.rag.data.corpus_registry import combine_registered_corpora
from app.rag.data.data_paths import (
    RERANK_CACHE_DIR,
    RESULT_INDEXES_DIR,
    RETRIEVAL_CACHE_DIR,
    RUNS_DIR,
)
from app.rag.data.json_utils import save_json_atomic
from app.rag.evaluation.eval_shared import (
    ConcurrencyConfig,
    EvaluationLLMConfig,
    build_eval_result,
    build_medical_eval_prompt,
    call_llm,
    create_eval_context,
)
from app.rag.experiments.phase1_formal_ablation import (
    EMBEDDING_PROVIDERS,
    FAISS_INDEX_TYPE,
    RETRIEVAL_CACHE_TOP_K,
    FormalRunSpec,
)
from app.rag.retriever.runtime_config import (
    DEFAULT_EMBEDDING_API_BASE_URL,
    first_env_value,
)


FORMAL_DEV_QUESTION_LIMIT: Optional[int] = None
EMBED_BATCH_SIZE = 128
LLM_PROGRESS_EVERY = 10
API_EMBEDDING_TIMEOUT = 120.0
API_EMBEDDING_MAX_RETRIES = 5
MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
MEDCPT_BATCH_SIZE = 16
MEDCPT_QUERY_MAX_LENGTH = 64
MEDCPT_ARTICLE_MAX_LENGTH = 512


@dataclass(frozen=True)
class DenseIndexPaths:
    """Shared paths for one corpus and embedding cache."""

    root: Path
    manifest: Path
    documents: Path
    chunk_embeddings: Path
    faiss_index: Path


@dataclass(frozen=True)
class FormalRunPaths:
    """All generated artifact paths for one executable formal run."""

    run_dir: Path
    retrieval_dir: Path
    rerank_dir: Path
    query_embeddings: Path
    retrieval_top80: Path
    rerank_outputs: Path
    final_prompts: Path
    llm_outputs: Path
    token_usage: Path
    estimated_token_cost: Path
    result_summary: Path


def _slug(value: Any) -> str:
    return str(value).lower().replace("/", "_").replace("\\", "_").replace(".", "p")


def _provider_by_model(model_name: str) -> Any:
    for provider in EMBEDDING_PROVIDERS:
        if provider.model == model_name:
            return provider
    raise KeyError(f"Unknown formal embedding model: {model_name}")


def dense_index_paths(run: FormalRunSpec) -> DenseIndexPaths:
    """Share dense indexes across k/alpha/reranker runs with the same inputs."""
    root = (
        RESULT_INDEXES_DIR
        / f"{_slug(run.corpus_version)}__{_slug(run.embedding_model)}__{FAISS_INDEX_TYPE}"
    )
    return DenseIndexPaths(
        root=root,
        manifest=root / "manifest.json",
        documents=root / "documents.jsonl",
        chunk_embeddings=root / "chunk_embeddings.npy",
        faiss_index=root / "faiss.index",
    )


def formal_run_paths(run: FormalRunSpec) -> FormalRunPaths:
    run_dir = RUNS_DIR / run.run_id
    return FormalRunPaths(
        run_dir=run_dir,
        retrieval_dir=RETRIEVAL_CACHE_DIR / run.run_id,
        rerank_dir=RERANK_CACHE_DIR / run.run_id,
        query_embeddings=RETRIEVAL_CACHE_DIR / run.run_id / "query_embeddings.npy",
        retrieval_top80=RETRIEVAL_CACHE_DIR / run.run_id / "retrieval_top80.jsonl",
        rerank_outputs=RERANK_CACHE_DIR / run.run_id / "rerank_outputs.jsonl",
        final_prompts=run_dir / "final_prompts.jsonl",
        llm_outputs=run_dir / "llm_outputs.jsonl",
        token_usage=run_dir / "token_usage.json",
        estimated_token_cost=run_dir / "estimated_token_cost.json",
        result_summary=run_dir / "result_summary.json",
    )


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype("float32")


def _api_embedding_client() -> OpenAI:
    api_key = first_env_value("RAG_EMBEDDING_API_KEY", "SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("RAG_EMBEDDING_API_KEY or SILICONFLOW_API_KEY is required")
    return OpenAI(
        api_key=api_key,
        base_url=first_env_value(
            "RAG_EMBEDDING_API_BASE_URL",
            default=DEFAULT_EMBEDDING_API_BASE_URL,
        ),
        timeout=API_EMBEDDING_TIMEOUT,
        max_retries=API_EMBEDDING_MAX_RETRIES,
    )


def _embed_api_texts(texts: Sequence[str], model_name: str) -> np.ndarray:
    client = _api_embedding_client()
    vectors: List[List[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = list(texts[start : start + EMBED_BATCH_SIZE])
        response = client.embeddings.create(model=model_name, input=batch)
        vectors.extend([item.embedding for item in response.data])
        print(
            f"  embedded {min(start + len(batch), len(texts)):,}/{len(texts):,} texts",
            flush=True,
        )
    return _normalize_rows(np.asarray(vectors, dtype="float32"))


def _embed_medcpt_texts(texts: Sequence[str], *, is_query: bool) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = MEDCPT_QUERY_MODEL if is_query else MEDCPT_ARTICLE_MODEL
    max_length = MEDCPT_QUERY_MAX_LENGTH if is_query else MEDCPT_ARTICLE_MAX_LENGTH
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    vectors: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), MEDCPT_BATCH_SIZE):
            batch = list(texts[start : start + MEDCPT_BATCH_SIZE])
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            embeds = model(**encoded).last_hidden_state[:, 0, :].detach().cpu().numpy()
            vectors.append(embeds.astype("float32"))
            print(
                f"  MedCPT embedded {min(start + len(batch), len(texts)):,}/{len(texts):,} texts",
                flush=True,
            )
    return _normalize_rows(np.vstack(vectors))


def embed_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    backend: str,
    is_query: bool,
) -> np.ndarray:
    """Single embedding dispatch point so backends cannot leak into main RAG code."""
    if backend == "siliconflow_api":
        return _embed_api_texts(texts, model_name)
    if backend == "local_medcpt":
        return _embed_medcpt_texts(texts, is_query=is_query)
    raise ValueError(f"Unsupported formal embedding backend: {backend}")


def load_corpus_documents(corpus_version: str) -> List[Dict[str, Any]]:
    """Load normalized corpus rows for the requested formal corpus variant."""
    if corpus_version == "statpearls":
        sources = ("statpearls",)
    elif corpus_version == "statpearls_textbooks":
        sources = ("statpearls", "textbooks")
    else:
        raise ValueError(f"Unknown formal corpus version: {corpus_version}")

    result = combine_registered_corpora(selected_sources=sources)
    records: List[Dict[str, Any]] = []
    for index, record in enumerate(result["records"]):
        text = str(record.get("contents") or record.get("content") or "").strip()
        if not text:
            continue
        records.append(
            {
                "doc_id": str(record.get("id") or f"{corpus_version}-{index}"),
                "title": record.get("title", ""),
                "source": record.get("source", "unknown"),
                "text": text,
                "metadata": {
                    key: value
                    for key, value in record.items()
                    if key not in {"content", "contents"}
                },
            }
        )
    if not records:
        raise ValueError(f"No documents loaded for corpus version {corpus_version}")
    return records


def ensure_dense_index(run: FormalRunSpec) -> Dict[str, Any]:
    """Build or load chunk embedding, document, and FAISS caches for one run."""
    paths = dense_index_paths(run)
    if paths.manifest.exists() and paths.faiss_index.exists() and paths.chunk_embeddings.exists():
        return json.loads(paths.manifest.read_text(encoding="utf-8"))

    provider = _provider_by_model(run.embedding_model)
    paths.root.mkdir(parents=True, exist_ok=True)
    documents = load_corpus_documents(run.corpus_version)
    _write_jsonl(paths.documents, documents)
    texts = [str(document["text"]) for document in documents]
    print(
        f"Building dense cache for corpus={run.corpus_version}, "
        f"embedding={run.embedding_model}, documents={len(texts):,}",
        flush=True,
    )
    started_at = time.time()
    embeddings = embed_texts(
        texts,
        model_name=provider.model,
        backend=provider.backend,
        is_query=False,
    )
    np.save(paths.chunk_embeddings, embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(paths.faiss_index))
    manifest = {
        "corpus_version": run.corpus_version,
        "document_count": len(documents),
        "embedding_model": run.embedding_model,
        "embedding_backend": provider.backend,
        "faiss_index_type": FAISS_INDEX_TYPE,
        "embedding_dim": int(embeddings.shape[1]),
        "documents_path": str(paths.documents),
        "chunk_embeddings_path": str(paths.chunk_embeddings),
        "faiss_index_path": str(paths.faiss_index),
        "build_time_seconds": time.time() - started_at,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    save_json_atomic(paths.manifest, manifest)
    return manifest


def retrieve_top80(
    run: FormalRunSpec,
    questions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    paths = formal_run_paths(run)
    if paths.retrieval_top80.exists() and paths.query_embeddings.exists():
        return list(_iter_jsonl(paths.retrieval_top80))

    index_manifest = ensure_dense_index(run)
    index_paths = dense_index_paths(run)
    documents = list(_iter_jsonl(index_paths.documents))
    index = faiss.read_index(str(index_paths.faiss_index))
    provider = _provider_by_model(run.embedding_model)
    query_texts = [str(item["question"]) for item in questions]
    query_embeddings = embed_texts(
        query_texts,
        model_name=provider.model,
        backend=provider.backend,
        is_query=True,
    )
    paths.retrieval_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths.query_embeddings, query_embeddings)

    limit = min(RETRIEVAL_CACHE_TOP_K, int(index.ntotal))
    scores, indices = index.search(query_embeddings, limit)
    rows: List[Dict[str, Any]] = []
    for question_index, item in enumerate(questions):
        contexts = []
        for rank, doc_index in enumerate(indices[question_index].tolist(), start=1):
            if doc_index < 0:
                continue
            document = documents[doc_index]
            contexts.append(
                {
                    "rank": rank,
                    "score": float(scores[question_index][rank - 1]),
                    "doc_id": document["doc_id"],
                    "source": document.get("source"),
                    "title": document.get("title"),
                    "text": document["text"],
                    "metadata": document.get("metadata", {}),
                }
            )
        rows.append(
            {
                "question_id": item.get("id", f"dev-{question_index + 1}"),
                "question": item["question"],
                "retrieval_top_k": limit,
                "index_manifest": index_manifest,
                "contexts": contexts,
            }
        )
    _write_jsonl(paths.retrieval_top80, rows)
    return rows


def _resolve_int_k(value: Any, default: int = 5) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default


def build_final_prompt(item: Mapping[str, Any], contexts: Sequence[Mapping[str, Any]]) -> str:
    context_text = "\n\n".join(
        f"[{index}] {context['text']}" for index, context in enumerate(contexts, start=1)
    )
    return build_medical_eval_prompt(
        question=str(item["question"]),
        options=item.get("options", []),
        context=context_text,
    )


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


async def execute_naive_run(
    run: FormalRunSpec,
    questions: Sequence[Dict[str, Any]],
    *,
    llm_config: Optional[EvaluationLLMConfig] = None,
) -> Dict[str, Any]:
    """Execute a dense-only formal run over the provided dev questions."""
    if run.pipeline != "naive_rag":
        raise ValueError(f"execute_naive_run only supports naive_rag, got {run.pipeline}")

    selected_questions = list(questions[:FORMAL_DEV_QUESTION_LIMIT] if FORMAL_DEV_QUESTION_LIMIT else questions)
    run_paths = formal_run_paths(run)
    for directory in (run_paths.run_dir, run_paths.retrieval_dir, run_paths.rerank_dir):
        directory.mkdir(parents=True, exist_ok=True)

    retrieval_rows = retrieve_top80(run, selected_questions)
    k = _resolve_int_k(run.k)
    llm = llm_config or EvaluationLLMConfig()
    ctx = create_eval_context(llm, ConcurrencyConfig())
    prompt_rows: List[Dict[str, Any]] = []
    llm_rows: List[Dict[str, Any]] = []
    detailed_results: List[Dict[str, Any]] = []
    correct = 0
    prompt_tokens = 0
    completion_tokens = 0
    started_at = time.time()

    for index, item in enumerate(selected_questions, start=1):
        retrieval = retrieval_rows[index - 1]
        contexts = retrieval["contexts"][:k]
        prompt = build_final_prompt(item, contexts)
        response = await call_llm(ctx, prompt)
        result = build_eval_result(
            item,
            response,
            {
                "retrieved_docs": len(contexts),
                "scores": [context["score"] for context in contexts],
                "contexts": [context["text"] for context in contexts],
            },
        )
        correct += 1 if result["is_correct"] else 0
        prompt_token_estimate = _estimate_tokens(prompt)
        completion_token_estimate = _estimate_tokens(response)
        prompt_tokens += prompt_token_estimate
        completion_tokens += completion_token_estimate
        prompt_rows.append(
            {
                "question_id": item.get("id", f"dev-{index}"),
                "prompt": prompt,
                "prompt_token_estimate": prompt_token_estimate,
            }
        )
        llm_rows.append(
            {
                "question_id": item.get("id", f"dev-{index}"),
                "response": response,
                "completion_token_estimate": completion_token_estimate,
                "predicted_answer": result["predicted_answer"],
                "correct_answer": result["correct_answer"],
                "is_correct": result["is_correct"],
            }
        )
        detailed_results.append(result)
        if index == 1 or index % LLM_PROGRESS_EVERY == 0 or index == len(selected_questions):
            print(
                f"  {run.run_id}: answered {index:,}/{len(selected_questions):,}, "
                f"accuracy={correct / index:.4f}",
                flush=True,
            )

    elapsed = time.time() - started_at
    token_usage = {
        "prompt_tokens_estimated": prompt_tokens,
        "completion_tokens_estimated": completion_tokens,
        "total_tokens_estimated": prompt_tokens + completion_tokens,
        "estimator": "char_count_div_4",
    }
    summary = {
        "run": asdict(run),
        "status": "completed",
        "total_questions": len(selected_questions),
        "correct": correct,
        "accuracy": correct / len(selected_questions) if selected_questions else 0.0,
        "latency_seconds": elapsed,
        "questions_per_second": len(selected_questions) / elapsed if elapsed > 0 else 0.0,
        "retrieval_time_seconds": None,
        "rerank_time_seconds": 0.0,
        "token_usage": token_usage,
        "artifact_paths": {key: str(value) for key, value in asdict(run_paths).items()},
        "detailed_results": detailed_results,
    }
    _write_jsonl(run_paths.final_prompts, prompt_rows)
    _write_jsonl(run_paths.llm_outputs, llm_rows)
    _write_jsonl(run_paths.rerank_outputs, [])
    save_json_atomic(run_paths.token_usage, token_usage)
    save_json_atomic(
        run_paths.estimated_token_cost,
        {
            "status": "not_priced",
            "reason": "Provider price table is not encoded in the experiment runner.",
            "token_usage": token_usage,
        },
    )
    save_json_atomic(run_paths.result_summary, summary)
    return summary

