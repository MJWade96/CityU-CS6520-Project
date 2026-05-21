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
from ..retriever.vector_store import MedicalVectorStore
from ..utils.progress_manager import EvaluationProgressManager
from .eval_shared import (
    ConcurrencyConfig,
    EvaluationLLMConfig,
    RateLimiter,
    build_eval_result,
    get_correct_answer_letter,
    load_questions,
    parse_optional_bool_env,
    split_questions,
)
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
        default_factory=lambda: os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-large")
    )
    reranker_device: str = field(
        default_factory=lambda: os.getenv("RAG_RERANKER_DEVICE", "auto")
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
    if config.use_hybrid_retrieval:
        hybrid = HybridRetriever.from_vector_store(
            vectorstore,
            similarity_top_k=config.top_k,
            use_async=True,
        )
        retriever = hybrid.fusion_retriever
    else:
        retriever = vectorstore.as_retriever(similarity_top_k=config.top_k)

    node_postprocessors = None
    if config.use_reranker:
        reranker = RerankerPipeline(
            use_cross_encoder=True,
            cross_encoder_model=config.reranker_model,
            cross_encoder_device=config.reranker_device,
            top_k=config.top_k,
        )
        if reranker.cross_encoder is not None and reranker.cross_encoder.available:
            node_postprocessors = [reranker.cross_encoder.model]

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=create_llm(config.llm),
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
    print(f"  Query Rewrite: {config.use_query_rewrite}")
    print(f"  LLM Query Rewrite: {config.use_llm_query_rewrite}")
    print(f"  LLM Query Rewrite Mode: {config.llm_query_rewrite_mode}")
    print(f"  Reranker: {config.use_reranker}")
    print(f"  Reranker Model: {config.reranker_model}")
    print(f"  Reranker Device: {config.reranker_device}")
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
    vectorstore = load_vector_store(config.vector_store_path)
    questions = load_questions(str(config.question_file))
    dev_set, test_set = split_questions(questions, config.dev_size, config.test_size)
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
        "evaluation_backend": run_names.evaluation_type,
        "use_hybrid_retrieval": config.use_hybrid_retrieval,
        "use_query_rewrite": config.use_query_rewrite,
        "use_llm_query_rewrite": config.use_llm_query_rewrite,
        "llm_query_rewrite_mode": config.llm_query_rewrite_mode,
        "use_reranker": config.use_reranker,
        "reranker_model": config.reranker_model,
        "reranker_device": config.reranker_device,
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
