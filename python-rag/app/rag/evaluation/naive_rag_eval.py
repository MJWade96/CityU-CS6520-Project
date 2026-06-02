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
    build_medical_eval_prompt,
    call_llm,
    build_eval_result,
    create_eval_context,
    format_options,
    format_retrieved_contexts,
    get_qwen_openai_like_kwargs,
    iter_pipeline_in_order,
    load_questions,
    question_id,
    serialize_document_candidates,
    serialize_node_candidates,
    split_questions,
    update_progress,
)
from app.rag.experiments.phase1_formal_ablation import LOCAL_EMBEDDING_BACKENDS
from app.rag.experiments.formal_cache_metadata import manifest_metadata, path_fingerprint

from .formal_local_embedding_adapter import LocalEmbeddingFormalRetriever
from . import formal_artifacts


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


def _formal_run_manifest(
    config: NaiveRAGEvalConfig,
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


async def _run_formal_naive_evaluation(
    config: NaiveRAGEvalConfig,
    questions: List[Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    """Formal Naive path: retrieve with question text, then call the shared LLM."""
    assert config.formal_run_id is not None
    metadata = config.formal_metadata or {}
    run_path = formal_artifacts.run_dir(config.formal_run_id)
    retrieval_path = formal_artifacts.retrieval_cache_dir(
        str(metadata.get("query_cache_id") or config.formal_run_id)
    )
    query_texts_path = retrieval_path / "query_texts.jsonl"
    retrieval_top10_path = retrieval_path / "retrieval_top10.jsonl"
    selected_contexts_path = run_path / "selected_contexts.jsonl"
    final_prompts_path = run_path / "final_prompts.jsonl"
    llm_outputs_path = run_path / "llm_outputs.jsonl"
    evaluation_outputs_path = run_path / "evaluation_outputs.jsonl"
    files = {
        "query_texts": str(query_texts_path),
        "retrieval_top10": str(retrieval_top10_path),
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

    local_embedding_retriever: Optional[LocalEmbeddingFormalRetriever] = None
    vectorstore: Optional[MedicalVectorStore] = None
    if metadata.get("embedding_backend") in LOCAL_EMBEDDING_BACKENDS:
        local_embedding_retriever = LocalEmbeddingFormalRetriever.load(
            corpus_version=str(metadata["corpus_version"]),
            index_root=config.vector_store_path,
            query_cache_id=str(metadata["query_cache_id"]),
        )
    else:
        vectorstore = load_vector_store(config.vector_store_path)

    ctx = create_eval_context(config.llm, config.concurrency)
    query_text_rows = formal_artifacts.rows_by_question_id(query_texts_path)
    retrieval_rows = formal_artifacts.rows_by_question_id(retrieval_top10_path)
    selected_rows = formal_artifacts.rows_by_question_id(selected_contexts_path)
    prompt_rows = formal_artifacts.rows_by_question_id(final_prompts_path)
    llm_rows = formal_artifacts.rows_by_question_id(llm_outputs_path)
    evaluation_rows = formal_artifacts.rows_by_question_id(evaluation_outputs_path)
    completed_ids = set(evaluation_rows)
    results: List[Dict[str, Any]] = [dict(row["result"]) for row in evaluation_rows.values()]
    correct = sum(1 for result in results if result.get("is_correct"))
    start_time = time.time()
    retrieval_top_k = max(10, top_k)
    generator_jobs: List[Dict[str, Any]] = []

    for index, item in enumerate(questions, start=1):
        current_question_id = question_id(item, index)
        if current_question_id in completed_ids:
            continue

        query_text = str(item["question"])
        if current_question_id not in query_text_rows:
            query_text_row = {
                "question_id": current_question_id,
                "question": query_text,
                "query_text": query_text,
                "query_text_source": "medqa_usmle_question_field",
                "contains_options": False,
                "contains_answer_prompt": False,
            }
            formal_artifacts.append_jsonl_with_checkpoint(query_texts_path, query_text_row)
            query_text_rows[current_question_id] = query_text_row

        if current_question_id in retrieval_rows:
            candidates = list(retrieval_rows[current_question_id]["candidates"])
        else:
            retrieval_started = time.time()
            if local_embedding_retriever is not None:
                retrieved = await asyncio.to_thread(
                    local_embedding_retriever.retrieve,
                    question_id=current_question_id,
                    query_text=query_text,
                    k=retrieval_top_k,
                )
                candidates = serialize_document_candidates(retrieved)
            else:
                assert vectorstore is not None
                nodes = await asyncio.to_thread(vectorstore.retrieve, query_text, retrieval_top_k)
                candidates = serialize_node_candidates(nodes)
            retrieval_elapsed = time.time() - retrieval_started
            retrieval_row = {
                "question_id": current_question_id,
                "query_text": query_text,
                "candidates": candidates,
                "retrieval_time_seconds": retrieval_elapsed,
            }
            formal_artifacts.append_jsonl_with_checkpoint(retrieval_top10_path, retrieval_row)
            retrieval_rows[current_question_id] = retrieval_row

        if current_question_id in selected_rows:
            selected = list(selected_rows[current_question_id]["selected_contexts"])
        else:
            selected = candidates[:top_k]
            selected_row = {"question_id": current_question_id, "selected_contexts": selected}
            formal_artifacts.append_jsonl_with_checkpoint(selected_contexts_path, selected_row)
            selected_rows[current_question_id] = selected_row

        if current_question_id in prompt_rows:
            prompt = str(prompt_rows[current_question_id]["prompt"])
        else:
            context = format_retrieved_contexts([candidate["text"] for candidate in selected])
            prompt = build_medical_eval_prompt(item["question"], item.get("options", []), context)
            prompt_row = {"question_id": current_question_id, "prompt": prompt}
            formal_artifacts.append_jsonl_with_checkpoint(final_prompts_path, prompt_row)
            prompt_rows[current_question_id] = prompt_row
        generator_jobs.append(
            {
                "question_id": current_question_id,
                "item": item,
                "prompt": prompt,
                "selected": selected,
                "response": llm_rows.get(current_question_id, {}).get("response"),
            }
        )

    async def generate_answer(
        _job_index: int,
        job: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        if job.get("response") is not None:
            response = str(job["response"])
        else:
            response = await call_llm(ctx, str(job["prompt"]))
        selected_candidates = list(job["selected"])
        result = build_eval_result(
            job["item"],
            response,
            {
                "retrieved_docs": len(selected_candidates),
                "scores": [candidate["score"] for candidate in selected_candidates],
                "contexts": [candidate["text"] for candidate in selected_candidates],
            },
        )
        return response, result

    async for _job_index, job, generated in iter_pipeline_in_order(
        generator_jobs,
        max_concurrent=config.concurrency.max_concurrent,
        worker=generate_answer,
    ):
        response, result = generated
        current_question_id = str(job["question_id"])
        if current_question_id not in llm_rows:
            llm_row = {"question_id": current_question_id, "response": response}
            formal_artifacts.append_jsonl_with_checkpoint(llm_outputs_path, llm_row)
            llm_rows[current_question_id] = llm_row
        results.append(result)
        if result["is_correct"]:
            correct += 1
        if current_question_id not in evaluation_rows:
            evaluation_row = {"question_id": current_question_id, "result": result}
            formal_artifacts.append_jsonl_with_checkpoint(evaluation_outputs_path, evaluation_row)
            evaluation_rows[current_question_id] = evaluation_row
            completed_ids.add(current_question_id)
        processed = len(results)
        if processed == len(questions) or processed % 5 == 0:
            print(
                f"[formal][{config.formal_run_id}] {processed}/{len(questions)} "
                f"acc={correct / processed:.4f}",
                flush=True,
            )

    elapsed = time.time() - start_time
    metrics = {
        "run_id": config.formal_run_id,
        "dataset_name": "Formal Dev Set",
        "top_k": top_k,
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
            "pipeline": "naive_rag",
            "processed_questions": len(results),
            "files": {
                "query_texts": str(query_texts_path),
                "retrieval_top10": str(retrieval_top10_path),
            },
            **manifest_metadata(
                key={
                    "cache_id": retrieval_path.name,
                    "embedding_model": metadata.get("embedding_model"),
                    "retrieval_top_k": retrieval_top_k,
                },
                input_artifacts={
                    "index_path": str(config.vector_store_path),
                    "query_embeddings_cache_id": metadata.get("query_cache_id"),
                },
                parameters={
                    "pipeline": "naive_rag",
                    "top_k": top_k,
                    "retrieval_top_k": retrieval_top_k,
                },
                dataset_split="dev",
                fingerprint={
                    "query_texts": path_fingerprint(query_texts_path),
                    "retrieval_top10": path_fingerprint(retrieval_top10_path),
                },
            ),
        },
    )
    return {
        "best_k": top_k,
        "dev_scores": {},
        "test_results": metrics,
        "recall_scores": {},
        "output_paths": {"run_dir": run_path},
    }


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

    async def evaluate_item(question_index: int, item: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_query(item["question"], item.get("options", []))
        async with semaphore:
            await rate_limiter.acquire()
            response = await query_engine.aquery(prompt)
        return build_eval_result(item, str(response), extract_rag_metadata(response))

    async for question_index, _item, result in iter_pipeline_in_order(
        remaining_questions,
        max_concurrent=batch_size,
        worker=evaluate_item,
        start_index=start_from,
    ):
        processed_questions = question_index + 1
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
    run_names = NAIVE_RAG_RUN_NAMES
    questions = load_questions(str(config.question_file))
    dev_set, test_set = split_questions(questions, config.dev_size, config.test_size)
    if config.formal_run_id is not None:
        if config.manual_top_k is None:
            raise ValueError("Formal Naive RAG requires a concrete manual_top_k")
        return await _run_formal_naive_evaluation(config, test_set, config.manual_top_k)

    vectorstore = load_vector_store(config.vector_store_path)

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
