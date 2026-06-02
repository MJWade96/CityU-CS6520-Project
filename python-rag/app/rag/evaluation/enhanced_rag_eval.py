"""Native enhanced RAG evaluation pipeline."""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.query_engine import RetrieverQueryEngine

from ..data.data_paths import EVALUATION_RESULTS_DIR, FAISS_INDEX_DIR, MEDQA_FILE
from ..retriever.hybrid_retriever import HybridRetriever
from ..retriever.query_rewrite import QueryRewritePipeline
from ..retriever.reranker import RerankerPipeline
from ..retriever.runtime_config import (
    DEFAULT_API_RERANKER_MODEL,
    DEFAULT_RERANKER_API_URL,
    first_env_value,
)
from ..retriever.vector_store import MedicalVectorStore
from app.rag.experiments.phase1_formal_ablation import LOCAL_EMBEDDING_BACKENDS
from ..utils.progress_manager import EvaluationProgressManager
from .eval_shared import (
    ConcurrencyConfig,
    EvaluationLLMConfig,
    RateLimiter,
    build_medical_eval_prompt,
    call_llm,
    create_eval_context,
    build_eval_result,
    format_retrieved_contexts,
    get_correct_answer_letter,
    load_questions,
    parse_optional_bool_env,
    question_id,
    serialize_node_candidates,
    split_questions,
)
from . import formal_artifacts
from .formal_local_rerank_cache import require_local_rerank_cache
from .formal_local_embedding_adapter import LocalEmbeddingFormalRetriever
from .naive_rag_eval import (
    build_query,
    create_llm,
    extract_rag_metadata,
    load_vector_store,
)


def _env_flag(name: str, default: bool) -> bool:
    value = parse_optional_bool_env(name, default=default)
    return default if value is None else value


@dataclass
class EnhancedEvaluationConfig:
    dev_size: int = 300
    test_size: Optional[int] = None
    top_k: int = 5
    retrieval_top_k: Optional[int] = None
    reranker_top_k: Optional[int] = None
    hybrid_alpha: float = 0.5
    vector_store_path: Path = FAISS_INDEX_DIR
    question_file: Path = MEDQA_FILE
    output_dir: Path = EVALUATION_RESULTS_DIR
    use_hybrid_retrieval: bool = True
    use_query_rewrite: bool = True
    use_llm_query_rewrite: bool = field(
        default_factory=lambda: _env_flag("RAG_ENHANCED_USE_LLM_QUERY_REWRITE", True)
    )
    llm_query_rewrite_mode: str = field(
        default_factory=lambda: os.getenv(
            "RAG_ENHANCED_LLM_QUERY_REWRITE_MODE",
            "auto",
        ).strip().lower()
    )
    llm_query_rewrite_auto_max_chars: int = field(
        default_factory=lambda: max(
            1,
            int(os.getenv("RAG_ENHANCED_LLM_QUERY_REWRITE_AUTO_MAX_CHARS", "160")),
        )
    )
    llm_query_rewrite_auto_max_words: int = field(
        default_factory=lambda: max(
            1,
            int(os.getenv("RAG_ENHANCED_LLM_QUERY_REWRITE_AUTO_MAX_WORDS", "24")),
        )
    )
    use_reranker: bool = True
    reranker_model: str = field(
        default_factory=lambda: os.getenv(
            "RAG_RERANKER_MODEL",
            DEFAULT_API_RERANKER_MODEL,
        )
    )
    reranker_api_url: str = field(
        default_factory=lambda: first_env_value(
            "RAG_RERANKER_API_URL",
            default=DEFAULT_RERANKER_API_URL,
        )
    )
    reranker_api_key: str = field(
        default_factory=lambda: first_env_value(
            "RAG_RERANKER_API_KEY",
            "SILICONFLOW_API_KEY",
        )
    )
    llm: EvaluationLLMConfig = field(default_factory=EvaluationLLMConfig)
    concurrency: ConcurrencyConfig = field(
        default_factory=lambda: ConcurrencyConfig(
            rpm_limit=int(
                os.getenv(
                    "RAG_ENHANCED_EVAL_RPM_LIMIT",
                    os.getenv("RAG_EVAL_RPM_LIMIT", "60"),
                )
            ),
            max_concurrent=int(
                os.getenv(
                    "RAG_ENHANCED_EVAL_MAX_CONCURRENT",
                    os.getenv("RAG_EVAL_MAX_CONCURRENT", "4"),
                )
            ),
        )
    )
    progress_save_every: int = field(
        default_factory=lambda: max(1, int(os.getenv("RAG_ENHANCED_EVAL_SAVE_EVERY", "5")))
    )
    progress_print_every: int = field(
        default_factory=lambda: max(
            1,
            int(os.getenv("RAG_ENHANCED_EVAL_PRINT_EVERY", "5")),
        )
    )
    heartbeat_enabled: bool = field(
        default_factory=lambda: _env_flag("RAG_ENHANCED_EVAL_HEARTBEAT_ENABLED", True)
    )
    heartbeat_interval_seconds: float = field(
        default_factory=lambda: max(
            1.0,
            float(os.getenv("RAG_ENHANCED_EVAL_HEARTBEAT_INTERVAL_SECONDS", "15")),
        )
    )
    question_start_log_enabled: bool = field(
        default_factory=lambda: _env_flag(
            "RAG_ENHANCED_EVAL_QUESTION_START_LOG_ENABLED",
            True,
        )
    )
    question_start_log_preview_chars: int = field(
        default_factory=lambda: max(
            20,
            int(os.getenv("RAG_ENHANCED_EVAL_QUESTION_START_LOG_PREVIEW_CHARS", "120")),
        )
    )
    formal_run_id: Optional[str] = None
    formal_metadata: Optional[Dict[str, Any]] = None

    @property
    def resolved_retrieval_top_k(self) -> int:
        """Use final top_k as the default retrieval depth for backward compatibility."""
        return self.retrieval_top_k if self.retrieval_top_k is not None else self.top_k

    @property
    def resolved_reranker_top_k(self) -> int:
        """Use final top_k as the default reranker output count."""
        return self.reranker_top_k if self.reranker_top_k is not None else self.top_k

    @property
    def dense_bm25_weights(self) -> tuple[float, float]:
        """Map alpha to QueryFusionRetriever weights without duplicating the formula."""
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ValueError(f"hybrid_alpha must be in [0, 1], got {self.hybrid_alpha}")
        return (self.hybrid_alpha, 1.0 - self.hybrid_alpha)


@dataclass(frozen=True)
class EnhancedEvaluationRunNames:
    artifact_prefix: str = "enhanced_rag_eval"
    run_name: str = "ENHANCED_RAG"
    evaluation_type: str = "ENHANCED_RAG"
    dev_script_name: str = "enhanced_eval_dev"
    test_script_name: str = "enhanced_eval_test"


def build_enhanced_query_engine(
    vectorstore: MedicalVectorStore,
    config: EnhancedEvaluationConfig,
) -> Any:
    llm = create_llm(config.llm)
    retrieval_top_k = config.resolved_retrieval_top_k
    reranker_top_k = config.resolved_reranker_top_k
    if config.use_hybrid_retrieval:
        hybrid = HybridRetriever.from_vector_store(
            vectorstore,
            llm=llm,
            similarity_top_k=retrieval_top_k,
            retriever_weights=config.dense_bm25_weights,
            use_async=True,
        )
        retriever = hybrid.fusion_retriever
    else:
        retriever = vectorstore.as_retriever(similarity_top_k=retrieval_top_k)

    node_postprocessors = None
    if config.use_reranker:
        reranker = RerankerPipeline(
            use_cross_encoder=True,
            cross_encoder_model=config.reranker_model,
            top_k=reranker_top_k,
            api_url=config.reranker_api_url,
            api_key=config.reranker_api_key,
        )
        if reranker.cross_encoder is not None and reranker.cross_encoder.available:
            node_postprocessors = [reranker.cross_encoder.model]

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        node_postprocessors=node_postprocessors,
        use_async=True,
    )


def should_use_llm_query_rewrite(
    query: str,
    query_rewriter: Optional[QueryRewritePipeline],
    config: EnhancedEvaluationConfig,
) -> bool:
    """Mirror the main enhanced-eval auto policy while keeping LlamaIndex native calls."""
    if not (config.use_query_rewrite and config.use_llm_query_rewrite):
        return False

    if query_rewriter is None or getattr(query_rewriter, "llm_rewriter", None) is None:
        return False

    mode = config.llm_query_rewrite_mode
    if mode == "always":
        return True
    if mode == "never":
        return False

    normalized = (query or "").strip()
    if not normalized:
        return False

    if any("\u4e00" <= ch <= "\u9fff" for ch in normalized):
        return True

    dict_rewriter = getattr(query_rewriter, "dict_rewriter", None)
    abbreviations = getattr(dict_rewriter, "ABBREVIATIONS", {})
    abbreviation_patterns = tuple(
        re.compile(rf"\b{re.escape(abbr)}\b", flags=re.IGNORECASE)
        for abbr in abbreviations
    )
    if any(pattern.search(normalized) for pattern in abbreviation_patterns):
        return True

    chinese_terms = getattr(dict_rewriter, "CHINESE_TERMS", {})
    if any(term in normalized for term in chinese_terms):
        return True

    if len(normalized) > config.llm_query_rewrite_auto_max_chars:
        return False

    if len(normalized.split()) > config.llm_query_rewrite_auto_max_words:
        return False

    return True


def _format_question_preview(text: str, max_chars: int) -> str:
    preview = re.sub(r"\s+", " ", (text or "")).strip()
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 3].rstrip() + "..."


def _build_progress_config(
    config: EnhancedEvaluationConfig,
    live_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Reuse one progress payload shape for checkpoints and live artifacts."""
    payload = dict(live_config or {})
    payload.update(
        {
            "top_k": config.top_k,
            "retrieval_top_k": config.resolved_retrieval_top_k,
            "reranker_top_k": config.resolved_reranker_top_k,
            "reranker_input_count": config.resolved_retrieval_top_k,
            "reranker_output_count": config.resolved_reranker_top_k,
            "reranker_backend": "api",
            "reranker_api_url": config.reranker_api_url,
            "hybrid_alpha": config.hybrid_alpha,
            "max_concurrent": config.concurrency.max_concurrent,
            "rpm_limit": config.concurrency.rpm_limit,
            "progress_save_every": config.progress_save_every,
            "progress_print_every": config.progress_print_every,
            "heartbeat_enabled": config.heartbeat_enabled,
            "heartbeat_interval_seconds": config.heartbeat_interval_seconds,
            "question_start_log_enabled": config.question_start_log_enabled,
            "question_start_log_preview_chars": config.question_start_log_preview_chars,
        }
    )
    return payload


def _formal_run_manifest(
    config: EnhancedEvaluationConfig,
    *,
    status: str,
    processed_questions: int,
    total_questions: int,
    accuracy: float,
    files: Dict[str, str],
) -> Dict[str, Any]:
    metadata = dict(config.formal_metadata or {})
    return {
        **metadata,
        "run_id": config.formal_run_id,
        "status": status,
        "processed_questions": processed_questions,
        "total_questions": total_questions,
        "accuracy": accuracy,
        "files": files,
    }


async def _run_formal_enhanced_evaluation(
    config: EnhancedEvaluationConfig,
    questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Formal Advanced path with explicit rewrite, retrieval, rerank, and LLM calls."""
    assert config.formal_run_id is not None
    metadata = config.formal_metadata or {}

    run_path = formal_artifacts.run_dir(config.formal_run_id)
    cache_id = str(metadata.get("query_cache_id") or config.formal_run_id)
    retrieval_path = formal_artifacts.retrieval_cache_dir(cache_id)
    rerank_path = formal_artifacts.rerank_cache_dir(cache_id)
    query_texts_path = retrieval_path / "query_texts.jsonl"
    dense_candidates_path = retrieval_path / "dense_candidates.jsonl"
    sparse_candidates_path = retrieval_path / "sparse_candidates.jsonl"
    fusion_candidates_path = retrieval_path / "fusion_candidates.jsonl"
    rerank_outputs_path = rerank_path / "rerank_outputs.jsonl"
    selected_contexts_path = run_path / "selected_contexts.jsonl"
    final_prompts_path = run_path / "final_prompts.jsonl"
    llm_outputs_path = run_path / "llm_outputs.jsonl"
    evaluation_outputs_path = run_path / "evaluation_outputs.jsonl"
    files = {
        "query_texts": str(query_texts_path),
        "dense_candidates": str(dense_candidates_path),
        "sparse_candidates": str(sparse_candidates_path),
        "fusion_candidates": str(fusion_candidates_path),
        "rerank_outputs": str(rerank_outputs_path),
        "selected_contexts": str(selected_contexts_path),
        "final_prompts": str(final_prompts_path),
        "llm_outputs": str(llm_outputs_path),
        "evaluation_outputs": str(evaluation_outputs_path),
    }
    formal_artifacts.write_run_manifest(
        config.formal_run_id,
        _formal_run_manifest(
            config,
            status="running",
            processed_questions=0,
            total_questions=len(questions),
            accuracy=0.0,
            files=files,
        ),
    )

    llm = create_llm(config.llm)
    local_embedding_retriever: Optional[LocalEmbeddingFormalRetriever] = None
    hybrid: Optional[HybridRetriever] = None
    if metadata.get("embedding_backend") in LOCAL_EMBEDDING_BACKENDS:
        local_embedding_retriever = LocalEmbeddingFormalRetriever.load(
            corpus_version=str(metadata["corpus_version"]),
            index_root=config.vector_store_path,
            query_cache_id=str(metadata["query_cache_id"]),
        )
    else:
        vectorstore = load_vector_store(config.vector_store_path)
        hybrid = HybridRetriever.from_vector_store(
            vectorstore,
            llm=llm,
            similarity_top_k=config.resolved_retrieval_top_k,
            retriever_weights=config.dense_bm25_weights,
            use_async=True,
        )
    query_rewriter = QueryRewritePipeline(
        use_dict=config.use_query_rewrite,
        use_llm=config.use_query_rewrite and config.use_llm_query_rewrite,
        llm_provider=config.llm.provider,
        llm_model=config.llm.model,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        llm_temperature=config.llm.temperature,
        llm_enable_thinking=config.llm.enable_thinking,
    )
    ctx = create_eval_context(config.llm, config.concurrency)
    completed_ids = formal_artifacts.completed_question_ids(evaluation_outputs_path)
    fusion_completed_ids = formal_artifacts.completed_question_ids(fusion_candidates_path)
    existing_results = formal_artifacts.load_jsonl(evaluation_outputs_path)
    results: List[Dict[str, Any]] = [dict(row["result"]) for row in existing_results]
    correct = sum(1 for result in results if result.get("is_correct"))
    start_time = time.time()
    expected_question_ids = [question_id(item, index) for index, item in enumerate(questions, start=1)]

    for index, item in enumerate(questions, start=1):
        current_question_id = question_id(item, index)
        if current_question_id in fusion_completed_ids:
            continue

        original_query = str(item["question"])
        rewrite_history: List[str] = []
        if local_embedding_retriever is not None:
            uses_llm_rewrite = False
            query_text = local_embedding_retriever.cached_query_text(current_question_id)
            query_text_source = "local_embedding_query_cache"
        else:
            uses_llm_rewrite = should_use_llm_query_rewrite(
                original_query,
                query_rewriter,
                config,
            )
            query_text = original_query
            query_text_source = "medqa_usmle_question_field"
        if config.use_query_rewrite and local_embedding_retriever is None:
            query_text, rewrite_history = await query_rewriter.arewrite(
                original_query,
                rate_limiter=ctx.rate_limiter,
                api_semaphore=ctx.semaphore,
                use_llm=uses_llm_rewrite,
            )
            query_text_source = "query_rewrite_pipeline"
        formal_artifacts.append_jsonl_if_question_missing(
            query_texts_path,
            {
                "question_id": current_question_id,
                "question": original_query,
                "original_query": original_query,
                "query_text": query_text,
                "query_text_source": query_text_source,
                "rewrite_metadata": {
                    "use_llm": uses_llm_rewrite,
                    "history": rewrite_history,
                },
                "contains_options": False,
                "contains_answer_prompt": False,
            },
        )

        retrieval_started = time.time()
        if local_embedding_retriever is not None:
            dense_nodes, sparse_nodes, fusion_nodes = await asyncio.to_thread(
                local_embedding_retriever.retrieve_components,
                question_id=current_question_id,
                query_text=query_text,
                k=config.resolved_retrieval_top_k,
                weights=config.dense_bm25_weights,
                llm=llm,
            )
        else:
            assert hybrid is not None
            dense_nodes, sparse_nodes, fusion_nodes = await asyncio.to_thread(
                hybrid.retrieve_components,
                query_text,
            )
        retrieval_elapsed = time.time() - retrieval_started
        dense_candidates = serialize_node_candidates(dense_nodes)
        sparse_candidates = serialize_node_candidates(sparse_nodes)
        fusion_candidates = serialize_node_candidates(fusion_nodes)
        for path, source, candidates in (
            (dense_candidates_path, "dense", dense_candidates),
            (sparse_candidates_path, "sparse_bm25", sparse_candidates),
            (fusion_candidates_path, "query_fusion", fusion_candidates),
        ):
            formal_artifacts.append_jsonl_with_checkpoint(
                path,
                {
                    "question_id": current_question_id,
                    "query_text": query_text,
                    "candidate_source": source,
                    "alpha": config.hybrid_alpha,
                    "top_k": config.resolved_retrieval_top_k,
                    "candidates": candidates,
                    "retrieval_time_seconds": retrieval_elapsed,
                },
            )

    formal_artifacts.write_json(
        retrieval_path / "manifest.json",
        {
            "cache_id": retrieval_path.name,
            "status": "completed",
            "pipeline": "advanced_rag",
            "processed_questions": len(expected_question_ids),
            "files": {
                "query_texts": str(query_texts_path),
                "dense_candidates": str(dense_candidates_path),
                "sparse_candidates": str(sparse_candidates_path),
                "fusion_candidates": str(fusion_candidates_path),
            },
        },
    )
    rerank_rows = require_local_rerank_cache(
        rerank_outputs_path,
        expected_question_ids=expected_question_ids,
        expected_model=config.reranker_model,
    )

    for index, item in enumerate(questions, start=1):
        current_question_id = question_id(item, index)
        if current_question_id in completed_ids:
            continue

        reranked_candidates = list(rerank_rows[current_question_id]["reranked_candidates"])
        selected = reranked_candidates[: config.resolved_reranker_top_k]
        formal_artifacts.append_jsonl_with_checkpoint(
            selected_contexts_path,
            {"question_id": current_question_id, "selected_contexts": selected},
        )
        context = format_retrieved_contexts([candidate["text"] for candidate in selected])
        prompt = build_medical_eval_prompt(item["question"], item.get("options", []), context)
        formal_artifacts.append_jsonl_with_checkpoint(
            final_prompts_path,
            {"question_id": current_question_id, "prompt": prompt},
        )
        response = await call_llm(ctx, prompt)
        formal_artifacts.append_jsonl_with_checkpoint(
            llm_outputs_path,
            {"question_id": current_question_id, "response": response},
        )
        result = build_eval_result(
            item,
            response,
            {
                "retrieved_docs": len(selected),
                "scores": [candidate["score"] for candidate in selected],
                "contexts": [candidate["text"] for candidate in selected],
            },
        )
        results.append(result)
        if result["is_correct"]:
            correct += 1
        formal_artifacts.append_jsonl_with_checkpoint(
            evaluation_outputs_path,
            {"question_id": current_question_id, "result": result},
        )
        processed = len(results)
        if processed == len(questions) or processed % config.progress_print_every == 0:
            print(
                f"[formal][{config.formal_run_id}] {processed}/{len(questions)} "
                f"acc={correct / processed:.4f}",
                flush=True,
            )

    elapsed = time.time() - start_time
    metrics = {
        "run_id": config.formal_run_id,
        "dataset_name": "Formal Dev Set",
        "top_k": config.top_k,
        "retrieval_top_k": config.resolved_retrieval_top_k,
        "reranker_top_k": config.resolved_reranker_top_k,
        "hybrid_alpha": config.hybrid_alpha,
        "total_questions": len(questions),
        "processed_questions": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0.0,
        "elapsed_time": elapsed,
        "questions_per_second": len(results) / elapsed if elapsed > 0 else 0.0,
        "detailed_results": results,
    }
    formal_artifacts.write_metrics(config.formal_run_id, metrics)
    formal_artifacts.write_run_manifest(
        config.formal_run_id,
        _formal_run_manifest(
            config,
            status="completed",
            processed_questions=len(results),
            total_questions=len(questions),
            accuracy=metrics["accuracy"],
            files={**files, "metrics": str(run_path / "metrics.json")},
        ),
    )
    formal_artifacts.write_json(
        retrieval_path / "manifest.json",
        {
            "cache_id": retrieval_path.name,
            "status": "completed",
            "pipeline": "advanced_rag",
            "processed_questions": len(results),
            "files": {
                "query_texts": str(query_texts_path),
                "dense_candidates": str(dense_candidates_path),
                "sparse_candidates": str(sparse_candidates_path),
                "fusion_candidates": str(fusion_candidates_path),
            },
        },
    )
    formal_artifacts.write_json(
        rerank_path / "manifest.json",
        {
            "cache_id": rerank_path.name,
            "status": "completed",
            "pipeline": "advanced_rag",
            "processed_questions": len(results),
            "files": {"rerank_outputs": str(rerank_outputs_path)},
        },
    )
    return {
        "dev_set_size": 0,
        "test_results": metrics,
        "output_paths": {"run_dir": run_path},
    }


async def evaluate_async_dataset(
    query_engine: Any,
    query_rewriter: Optional[QueryRewritePipeline],
    questions: List[Dict[str, Any]],
    config: EnhancedEvaluationConfig,
    *,
    run_names: EnhancedEvaluationRunNames,
    progress_mgr: Optional[EvaluationProgressManager] = None,
    artifact_paths: Optional[Dict[str, Path]] = None,
    live_config: Optional[Dict[str, Any]] = None,
    dataset_name: str = "Test Set",
    start_from: int = 0,
    initial_results: Optional[List[Dict[str, Any]]] = None,
    initial_correct: int = 0,
    initial_total: int = 0,
    initial_elapsed: float = 0.0,
) -> Dict[str, Any]:
    start_time = time.time() - initial_elapsed
    results: List[Dict[str, Any]] = list(initial_results or [])
    correct = initial_correct
    total = initial_total or len(results)
    remaining_questions = questions[start_from:]
    batch_size = max(1, config.concurrency.max_concurrent)
    persist_every = max(1, config.progress_save_every)
    print_every = max(1, config.progress_print_every)
    heartbeat_interval = max(1.0, config.heartbeat_interval_seconds)
    semaphore = asyncio.Semaphore(batch_size)
    rate_limiter = RateLimiter(
        requests_per_second=config.concurrency.requests_per_second,
        burst=batch_size,
    )
    progress_config = _build_progress_config(config, live_config)
    last_heartbeat_at = time.time()

    if config.heartbeat_enabled:
        print(f"Heartbeat enabled: every {heartbeat_interval:.0f}s")

    def emit_heartbeat(reason: str) -> None:
        nonlocal last_heartbeat_at

        if not config.heartbeat_enabled:
            return

        now = time.time()
        print(
            "  heartbeat: "
            f"committed={total}/{len(questions)}, "
            f"elapsed={now - start_time:.1f}s, "
            f"reason={reason}"
        )
        last_heartbeat_at = now

    async def evaluate_item(question_index: int, item: Dict[str, Any]) -> Dict[str, Any]:
        original_question = item.get("question", "")
        options = item.get("options", [])
        uses_llm_rewrite = should_use_llm_query_rewrite(
            original_question,
            query_rewriter,
            config,
        )

        if config.question_start_log_enabled:
            print(
                "  start: "
                f"q={question_index + 1}/{len(questions)}, "
                f"chars={len(original_question)}, "
                f"options={len(options)}, "
                f"hybrid={config.use_hybrid_retrieval}, "
                f"rewrite={config.use_query_rewrite}, "
                f"llm_rewrite={uses_llm_rewrite}, "
                f"reranker={config.use_reranker}, "
                f"top_k={config.top_k}, "
                f"retrieval_top_k={config.resolved_retrieval_top_k}, "
                f"reranker_top_k={config.resolved_reranker_top_k}, "
                f"preview=\"{_format_question_preview(original_question, config.question_start_log_preview_chars)}\""
            )

        question = original_question
        if config.use_query_rewrite and query_rewriter is not None:
            question, _ = await query_rewriter.arewrite(
                question,
                rate_limiter=rate_limiter,
                api_semaphore=semaphore,
                use_llm=uses_llm_rewrite,
            )

        prompt = build_query(question, options)
        async with semaphore:
            await rate_limiter.acquire()
            response = await query_engine.aquery(prompt)
        return build_eval_result(item, str(response), extract_rag_metadata(response))

    def persist_progress(processed_questions: int, error_message: Optional[str]) -> None:
        if not progress_mgr:
            return

        elapsed = time.time() - start_time
        should_print = (
            processed_questions == len(questions)
            or error_message is not None
            or processed_questions % print_every == 0
        )
        if should_print:
            progress_mgr.print_progress(
                run_name=run_names.run_name,
                dataset_name=dataset_name,
                processed_questions=processed_questions,
                total_questions=len(questions),
                correct_count=correct,
                elapsed_time=elapsed,
            )

        should_persist = (
            processed_questions == len(questions)
            or error_message is not None
            or processed_questions % persist_every == 0
        )
        if not should_persist:
            return

        progress_mgr.save_checkpoint(
            dataset_name=dataset_name,
            total_questions=len(questions),
            processed_questions=processed_questions,
            current_top_k=config.top_k,
            results=results,
            correct_count=correct,
            total_count=total,
            elapsed_time=elapsed,
            config=progress_config,
            script_name=run_names.test_script_name,
            error_message=error_message,
        )
        if artifact_paths and live_config:
            stage_result = progress_mgr.build_stage_result(
                dataset_name=dataset_name,
                total_questions=len(questions),
                processed_questions=processed_questions,
                correct_count=correct,
                elapsed_time=elapsed,
                detailed_results=results,
                top_k=config.top_k,
            )
            progress_mgr.write_live_results(
                artifact_paths=artifact_paths,
                run_name=run_names.run_name,
                evaluation_type=run_names.evaluation_type,
                config=live_config,
                stage_result=stage_result,
            )

    for batch_start in range(0, len(remaining_questions), batch_size):
        batch = remaining_questions[batch_start : batch_start + batch_size]
        batch_tasks = [
            asyncio.create_task(evaluate_item(start_from + batch_start + offset, item))
            for offset, item in enumerate(batch)
        ]
        pending_tasks = list(batch_tasks)

        while pending_tasks:
            done, pending = await asyncio.wait(
                pending_tasks,
                timeout=heartbeat_interval if config.heartbeat_enabled else None,
                return_when=asyncio.ALL_COMPLETED,
            )
            if not done:
                emit_heartbeat("awaiting_batch")
                continue
            pending_tasks = list(pending)

        for offset, item in enumerate(batch):
            processed_questions = start_from + batch_start + offset + 1
            task = batch_tasks[offset]
            try:
                evaluation_result = await task
            except Exception as exc:
                evaluation_result = {
                    "question": item.get("question", ""),
                    "options": item.get("options", []),
                    "correct_answer": get_correct_answer_letter(item),
                    "predicted_answer": None,
                    "is_correct": False,
                    "response": f"Error generating answer: {exc}",
                    "retrieved_docs": 0,
                    "error": str(exc),
                }
            if evaluation_result.get("is_correct"):
                correct += 1
            total += 1
            results.append(evaluation_result)

            if evaluation_result.get("error"):
                print(f"  ERROR on question {processed_questions}: {evaluation_result['error']}")

            persist_progress(processed_questions, evaluation_result.get("error"))

    elapsed = time.time() - start_time
    return {
        "dataset_name": dataset_name,
        "top_k": config.top_k,
        "retrieval_top_k": config.resolved_retrieval_top_k,
        "reranker_top_k": config.resolved_reranker_top_k,
        "hybrid_alpha": config.hybrid_alpha,
        "total_questions": len(questions),
        "processed_questions": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "elapsed_time": elapsed,
        "questions_per_second": total / elapsed if elapsed > 0 else 0.0,
        "detailed_results": results,
    }


def print_evaluation_header(
    config: EnhancedEvaluationConfig,
    dev_set: List[Dict[str, Any]],
    test_set: List[Dict[str, Any]],
) -> None:
    print("=" * 60)
    print("Enhanced Medical RAG System - Complete Evaluation")
    print("Phase 1 + Phase 2 Optimizations")
    print("=" * 60)
    print("\nEvaluation Scope:")
    print("  Only evaluating test set")
    print(f"  Dev set size (aligned, not evaluated here): {len(dev_set)} questions")
    print(f"  Test set: {len(test_set)} questions")
    print("\nInitializing Enhanced RAG Pipeline...")
    print(f"  Hybrid Retrieval: {config.use_hybrid_retrieval}")
    print(f"  Hybrid Alpha (dense weight): {config.hybrid_alpha}")
    print(f"  Retrieval Top K: {config.resolved_retrieval_top_k}")
    print(f"  Query Rewrite: {config.use_query_rewrite}")
    print(f"  LLM Query Rewrite: {config.use_llm_query_rewrite}")
    print(f"  LLM Query Rewrite Mode: {config.llm_query_rewrite_mode}")
    print(f"  Reranker: {config.use_reranker}")
    print("  Reranker Backend: api")
    print(f"  Reranker Top K: {config.resolved_reranker_top_k}")
    print(f"  Reranker Model: {config.reranker_model}")
    print(f"  Reranker API URL: {config.reranker_api_url}")
    print(f"  Max Concurrent: {config.concurrency.max_concurrent}")
    print(f"  Progress Save Every: {config.progress_save_every} questions")
    print(f"  Progress Print Every: {config.progress_print_every} questions")
    print(f"  Heartbeat Enabled: {config.heartbeat_enabled}")
    print(f"  Heartbeat Interval: {config.heartbeat_interval_seconds:.0f}s")
    print(f"  Question Start Logs: {config.question_start_log_enabled}")
    print(f"  Question Start Preview Chars: {config.question_start_log_preview_chars}")


def print_evaluation_summary(
    test_results: Dict[str, Any],
    output_paths: Dict[str, Path],
) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("\nFinal Results:")
    print(f"  Test Set Accuracy: {test_results['accuracy']:.4f}")
    print(
        f"\nEvaluation Time: {test_results['elapsed_time']:.1f}s "
        f"({test_results['questions_per_second']:.2f} questions/second)"
    )
    print("\n" + "=" * 60)
    print("Optimization Summary:")
    print("=" * 60)
    print("Active stack: Hybrid Retrieval, Query Rewrite, Cross-Encoder Reranking")
    print(f"Results JSON: {output_paths['json']}")
    print(f"Summary TXT: {output_paths['summary']}")
    print("=" * 60)


async def run_enhanced_evaluation(config: EnhancedEvaluationConfig) -> Dict[str, Any]:
    questions = load_questions(str(config.question_file))
    dev_set, test_set = split_questions(questions, config.dev_size, config.test_size)
    if config.formal_run_id is not None:
        return await _run_formal_enhanced_evaluation(config, test_set)

    vectorstore = load_vector_store(config.vector_store_path)
    run_names = EnhancedEvaluationRunNames()
    print_evaluation_header(config, dev_set, test_set)
    query_engine = build_enhanced_query_engine(vectorstore, config)
    print("[OK] Enhanced RAG Pipeline initialized")
    query_rewriter = QueryRewritePipeline(
        use_dict=config.use_query_rewrite,
        use_llm=config.use_query_rewrite and config.use_llm_query_rewrite,
        llm_provider=config.llm.provider,
        llm_model=config.llm.model,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        llm_temperature=config.llm.temperature,
        llm_enable_thinking=config.llm.enable_thinking,
    )
    progress_mgr = EvaluationProgressManager(output_dir=str(config.output_dir))
    artifact_paths = progress_mgr.create_run_artifacts(run_names.artifact_prefix)
    live_config = {
        "dev_set_size": len(dev_set),
        "test_set_size": len(test_set),
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.model,
        "vector_store": str(config.vector_store_path),
        "top_k": config.top_k,
        "retrieval_top_k": config.resolved_retrieval_top_k,
        "reranker_top_k": config.resolved_reranker_top_k,
        "reranker_input_count": config.resolved_retrieval_top_k,
        "reranker_output_count": config.resolved_reranker_top_k,
        "evaluation_backend": run_names.evaluation_type,
        "use_hybrid_retrieval": config.use_hybrid_retrieval,
        "hybrid_alpha": config.hybrid_alpha,
        "hybrid_retriever_weights": list(config.dense_bm25_weights),
        "use_query_rewrite": config.use_query_rewrite,
        "use_llm_query_rewrite": config.use_llm_query_rewrite,
        "llm_query_rewrite_mode": config.llm_query_rewrite_mode,
        "use_reranker": config.use_reranker,
        "reranker_backend": "api",
        "reranker_model": config.reranker_model,
        "reranker_api_url": config.reranker_api_url,
        "max_concurrent": config.concurrency.max_concurrent,
        "rpm_limit": config.concurrency.rpm_limit,
        "progress_save_every": config.progress_save_every,
        "progress_print_every": config.progress_print_every,
        "heartbeat_enabled": config.heartbeat_enabled,
        "heartbeat_interval_seconds": config.heartbeat_interval_seconds,
        "question_start_log_enabled": config.question_start_log_enabled,
        "question_start_log_preview_chars": config.question_start_log_preview_chars,
    }

    resume_test = progress_mgr.should_resume(run_names.test_script_name)
    resume_info_test = (
        progress_mgr.get_resume_info(run_names.test_script_name) if resume_test else None
    )
    test_results = await evaluate_async_dataset(
        query_engine=query_engine,
        query_rewriter=query_rewriter,
        questions=test_set,
        config=config,
        run_names=run_names,
        progress_mgr=progress_mgr,
        artifact_paths=artifact_paths,
        live_config=live_config,
        start_from=resume_info_test["start_from"] if resume_info_test else 0,
        initial_results=resume_info_test["results"] if resume_info_test else None,
        initial_correct=resume_info_test["correct_count"] if resume_info_test else 0,
        initial_total=resume_info_test["total_count"] if resume_info_test else 0,
        initial_elapsed=resume_info_test["elapsed_time"] if resume_info_test else 0.0,
    )
    progress_mgr.clear_checkpoint(run_names.test_script_name)
    output_paths = progress_mgr.write_final_results(
        artifact_paths=artifact_paths,
        run_name=run_names.run_name,
        evaluation_type=run_names.evaluation_type,
        config=live_config,
        stage_results={"test_set_evaluation": test_results},
        extra_sections=None,
    )
    print_evaluation_summary(test_results, output_paths)
    return {
        "dev_set_size": len(dev_set),
        "test_results": test_results,
        "output_paths": output_paths,
    }
