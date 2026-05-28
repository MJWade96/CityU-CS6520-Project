"""Primary native RAG evaluation pipeline."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llama_index.llms.openai_like import OpenAILike

from ..retriever.runtime_config import resolve_embedding_runtime
from ..retriever.vector_store import MedicalVectorStore
from ..utils.progress_manager import EvaluationProgressManager
from .config import EvaluationRunNames, NAIVE_RAG_RUN_NAMES, NaiveRAGEvalConfig
from .eval_shared import (
    EvaluationLLMConfig,
    RateLimiter,
    build_eval_result,
    format_options,
    get_qwen_openai_like_kwargs,
    load_questions,
    split_questions,
    update_progress,
)


def build_query(question: str, options: List[str]) -> str:
    """Keep evaluation prompt structure stable while delegating answering to the query engine."""
    return "\n".join(
        [
            "You are a medical expert assistant.",
            "Use the retrieved medical context to answer the multiple-choice question.",
            f"Question: {question}",
            "",
            "Options:",
            format_options(options),
            "",
            "Provide only the final answer in the following format:",
            "Answer: [A/B/C/D/E]",
        ]
    )


def create_llm(config: EvaluationLLMConfig) -> OpenAILike:
    """Create the native OpenAI-compatible LLM used by the query engine."""
    return OpenAILike(**get_qwen_openai_like_kwargs(config))


def extract_rag_metadata(response: Any) -> Dict[str, Any]:
    """Convert native source nodes into the shared evaluation payload."""
    source_nodes = list(getattr(response, "source_nodes", []) or [])
    return {
        "retrieved_docs": len(source_nodes),
        "scores": [float(node.score or 0.0) for node in source_nodes],
        "contexts": [node.node.get_content() for node in source_nodes],
    }


def load_vector_store(index_path: Path) -> MedicalVectorStore:
    """Load the persisted native FAISS store."""
    runtime = resolve_embedding_runtime(str(index_path), default_model="BAAI/bge-m3")
    vectorstore = MedicalVectorStore(
        embedding_model_name=runtime["model_name"],
        normalize_embeddings=True,
        embedding_api_base_url=runtime["api_base_url"],
        embedding_api_key=runtime["api_key"],
        embedding_api_dimensions=runtime["api_dimensions"],
        embedding_api_timeout=runtime["api_timeout"],
        embedding_api_max_retries=runtime["api_max_retries"],
    )
    vectorstore.load(str(index_path))
    return vectorstore


def evaluate_sync_dataset(
    vectorstore: MedicalVectorStore,
    llm_config: EvaluationLLMConfig,
    questions: List[Dict[str, Any]],
    top_k: int,
    *,
    run_name: str,
    evaluation_type: str,
    progress_mgr: Optional[EvaluationProgressManager] = None,
    artifact_paths: Optional[Dict[str, Path]] = None,
    live_config: Optional[Dict[str, Any]] = None,
    extra_sections: Optional[Dict[str, Any]] = None,
    dataset_name: str = "Development Set",
    script_name: str = "complete_eval_dev",
) -> Dict[str, Any]:
    """Evaluate a dataset synchronously with the native query engine."""
    query_engine = vectorstore.as_query_engine(
        llm=create_llm(llm_config),
        similarity_top_k=top_k,
    )
    start_time = time.time()
    results: List[Dict[str, Any]] = []
    correct = 0

    for index, item in enumerate(questions, start=1):
        response = query_engine.query(build_query(item["question"], item.get("options", [])))
        result = build_eval_result(item, str(response), extract_rag_metadata(response))
        results.append(result)
        if result["is_correct"]:
            correct += 1

        if progress_mgr:
            update_progress(
                progress_mgr=progress_mgr,
                artifact_paths=artifact_paths,
                live_config=live_config,
                extra_sections=extra_sections,
                dataset_name=dataset_name,
                total_questions=len(questions),
                processed_questions=index,
                correct_count=correct,
                elapsed=time.time() - start_time,
                results=results,
                run_name=run_name,
                evaluation_type=evaluation_type,
                config_payload={"top_k": top_k},
                script_name=script_name,
                top_k=top_k,
            )

    elapsed = time.time() - start_time
    return {
        "dataset_name": dataset_name,
        "top_k": top_k,
        "total_questions": len(questions),
        "processed_questions": len(questions),
        "correct": correct,
        "accuracy": correct / len(questions) if questions else 0.0,
        "elapsed_time": elapsed,
        "questions_per_second": len(questions) / elapsed if elapsed > 0 else 0.0,
        "detailed_results": results,
    }


async def evaluate_async_dataset(
    vectorstore: MedicalVectorStore,
    llm_config: EvaluationLLMConfig,
    questions: List[Dict[str, Any]],
    top_k: int,
    *,
    run_name: str,
    evaluation_type: str,
    max_concurrent: int,
    requests_per_second: float,
    progress_mgr: Optional[EvaluationProgressManager] = None,
    artifact_paths: Optional[Dict[str, Path]] = None,
    live_config: Optional[Dict[str, Any]] = None,
    extra_sections: Optional[Dict[str, Any]] = None,
    dataset_name: str = "Test Set",
    script_name: str = "complete_eval_test",
    start_from: int = 0,
    initial_results: Optional[List[Dict[str, Any]]] = None,
    initial_correct: int = 0,
    initial_elapsed: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate a dataset asynchronously with the native query engine."""
    query_engine = vectorstore.as_query_engine(
        llm=create_llm(llm_config),
        similarity_top_k=top_k,
    )
    semaphore = asyncio.Semaphore(max(1, max_concurrent))
    rate_limiter = RateLimiter(requests_per_second=requests_per_second, burst=max_concurrent)
    start_time = time.time() - initial_elapsed
    results: List[Dict[str, Any]] = list(initial_results or [])
    correct = initial_correct
    remaining_questions = questions[start_from:]
    batch_size = max(1, max_concurrent)

    async def evaluate_item(item: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_query(item["question"], item.get("options", []))
        async with semaphore:
            await rate_limiter.acquire()
            response = await query_engine.aquery(prompt)
        return build_eval_result(item, str(response), extract_rag_metadata(response))

    for batch_start in range(0, len(remaining_questions), batch_size):
        batch = remaining_questions[batch_start : batch_start + batch_size]
        batch_results = await asyncio.gather(*(evaluate_item(item) for item in batch))

        for offset, result in enumerate(batch_results, start=1):
            processed_questions = start_from + batch_start + offset
            results.append(result)
            if result["is_correct"]:
                correct += 1

            if progress_mgr:
                update_progress(
                    progress_mgr=progress_mgr,
                    artifact_paths=artifact_paths,
                    live_config=live_config,
                    extra_sections=extra_sections,
                    dataset_name=dataset_name,
                    total_questions=len(questions),
                    processed_questions=processed_questions,
                    correct_count=correct,
                    elapsed=time.time() - start_time,
                    results=results,
                    run_name=run_name,
                    evaluation_type=evaluation_type,
                    config_payload={"top_k": top_k},
                    script_name=script_name,
                    top_k=top_k,
                )

    elapsed = time.time() - start_time
    return {
        "dataset_name": dataset_name,
        "top_k": top_k,
        "total_questions": len(questions),
        "processed_questions": len(questions),
        "correct": correct,
        "accuracy": correct / len(questions) if questions else 0.0,
        "elapsed_time": elapsed,
        "questions_per_second": len(questions) / elapsed if elapsed > 0 else 0.0,
        "detailed_results": results,
    }


def find_best_top_k(
    vectorstore: MedicalVectorStore,
    dev_set: List[Dict[str, Any]],
    config: NaiveRAGEvalConfig,
    run_names: EvaluationRunNames,
    progress_mgr: Optional[EvaluationProgressManager] = None,
    artifact_paths: Optional[Dict[str, Path]] = None,
    live_config: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[int, float], Dict[str, Any]]:
    """Search for the best top-k on the development slice."""
    scores: Dict[int, float] = {}
    results_by_k: Dict[int, Dict[str, Any]] = {}

    for k in config.top_k_values:
        result = evaluate_sync_dataset(
            vectorstore=vectorstore,
            llm_config=config.llm,
            questions=dev_set,
            top_k=k,
            run_name=run_names.run_name,
            evaluation_type=run_names.evaluation_type,
            progress_mgr=progress_mgr,
            artifact_paths=artifact_paths,
            live_config=live_config,
            extra_sections={
                "hyperparameter_search": {
                    "k_values_tested": config.top_k_values,
                    "development_set_accuracy": scores,
                    "best_k": None,
                    "used_manual_top_k": False,
                },
            },
            dataset_name=f"Development Set (k={k})",
            script_name=run_names.dev_script_name,
        )
        scores[k] = result["accuracy"]
        results_by_k[k] = result

    best_k = max(scores, key=scores.get)
    return best_k, scores, results_by_k[best_k]


def calculate_recall_at_k(
    vectorstore: MedicalVectorStore,
    questions: List[Dict[str, Any]],
    k_values: List[int],
) -> Dict[int, float]:
    """Compute a simple answer-string recall@k metric from native retrieval results."""
    recall_scores: Dict[int, float] = {}
    for k in k_values:
        hits = 0
        for item in questions:
            answer = str(item.get("answer", "")).lower()
            source_nodes = vectorstore.retrieve(item["question"], k=k)
            if any(
                answer and answer in node_with_score.node.get_content().lower()
                for node_with_score in source_nodes
            ):
                hits += 1
        recall_scores[k] = hits / len(questions) if questions else 0.0
    return recall_scores


async def run_complete_evaluation(config: NaiveRAGEvalConfig) -> Dict[str, Any]:
    """Execute the complete native RAG flow behind the primary entrypoint names."""
    vectorstore = load_vector_store(config.vector_store_path)
    run_names = NAIVE_RAG_RUN_NAMES
    questions = load_questions(str(config.question_file))
    dev_set, test_set = split_questions(questions, config.dev_size, config.test_size)

    progress_mgr = EvaluationProgressManager(output_dir=str(config.output_dir))
    artifact_paths = progress_mgr.create_run_artifacts(run_names.artifact_prefix)
    live_config = {
        "dev_set_size": len(dev_set),
        "test_set_size": len(test_set),
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.model,
        "vector_store": str(config.vector_store_path),
        "manual_top_k": config.manual_top_k,
        "evaluation_backend": run_names.evaluation_type,
    }

    if config.manual_top_k is None:
        best_k, dev_scores, dev_results = find_best_top_k(
            vectorstore,
            dev_set,
            config,
            run_names,
            progress_mgr,
            artifact_paths,
            live_config,
        )
    else:
        best_k, dev_scores = config.manual_top_k, {}
        dev_results = {
            "dataset_name": "Development Set",
            "top_k": best_k,
            "total_questions": 0,
            "correct": 0,
            "accuracy": 0.0,
            "elapsed_time": 0.0,
            "questions_per_second": 0.0,
            "detailed_results": [],
        }

    resume_test = progress_mgr.should_resume(run_names.test_script_name)
    resume_info_test = (
        progress_mgr.get_resume_info(run_names.test_script_name) if resume_test else None
    )

    test_results = await evaluate_async_dataset(
        vectorstore=vectorstore,
        llm_config=config.llm,
        questions=test_set,
        top_k=best_k,
        run_name=run_names.run_name,
        evaluation_type=run_names.evaluation_type,
        max_concurrent=config.concurrency.max_concurrent,
        requests_per_second=config.concurrency.requests_per_second,
        progress_mgr=progress_mgr,
        artifact_paths=artifact_paths,
        live_config=live_config,
        extra_sections={
            "development_set_evaluation": dev_results,
            "hyperparameter_search": {
                "k_values_tested": (
                    config.top_k_values if config.manual_top_k is None else "manual"
                ),
                "development_set_accuracy": dev_scores,
                "best_k": best_k,
                "used_manual_top_k": config.manual_top_k is not None,
            },
        },
        script_name=run_names.test_script_name,
        start_from=resume_info_test["start_from"] if resume_info_test else 0,
        initial_results=resume_info_test["results"] if resume_info_test else None,
        initial_correct=resume_info_test["correct_count"] if resume_info_test else 0,
        initial_elapsed=resume_info_test["elapsed_time"] if resume_info_test else 0.0,
    )
    recall_scores = calculate_recall_at_k(vectorstore, test_set, [1, 3, 5, 10])
    progress_mgr.clear_checkpoint(run_names.test_script_name)
    paths = progress_mgr.write_final_results(
        artifact_paths=artifact_paths,
        run_name=run_names.run_name,
        evaluation_type=run_names.evaluation_type,
        config=live_config,
        stage_results={
            "development_set_evaluation": dev_results,
            "test_set_evaluation": test_results,
        },
        extra_sections={
            "hyperparameter_search": {
                "k_values_tested": (
                    config.top_k_values if config.manual_top_k is None else "manual"
                ),
                "development_set_accuracy": dev_scores,
                "best_k": best_k,
                "used_manual_top_k": config.manual_top_k is not None,
            },
            "retrieval_recall_at_k": recall_scores,
        },
    )

    return {
        "best_k": best_k,
        "dev_scores": dev_scores,
        "test_results": test_results,
        "recall_scores": recall_scores,
        "output_paths": paths,
    }


DEFAULT_NAIVE_RAG_CONFIG = NaiveRAGEvalConfig()
