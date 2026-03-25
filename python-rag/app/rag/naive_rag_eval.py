"""
Naive RAG evaluation pipeline.

The core evaluation logic lives here so ``complete_eval.py`` can stay as a thin
entrypoint and other scripts can reuse the same retrieval/generation flow.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI

from .data_paths import EVALUATION_DIR, EVALUATION_RESULTS_DIR, FAISS_INDEX_DIR
from .embeddings import get_langchain_embeddings
from .eval_shared import (
    ConcurrencyConfig,
    EvaluationLLMConfig,
    RateLimiter,
    build_medical_eval_prompt,
    create_async_client,
    extract_answer,
    get_qwen_completion_kwargs,
    get_qwen_langchain_kwargs,
    get_correct_answer_letter,
    load_questions,
    split_questions,
)
from .progress_manager import EvaluationProgressManager
from .vector_store import MedicalVectorStore


@dataclass
class NaiveRAGEvalConfig:
    dev_size: int = 50
    test_size: int = 50
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    manual_top_k: Optional[int] = 3
    vector_store_path: Path = FAISS_INDEX_DIR
    question_file: Path = EVALUATION_DIR / "medqa.json"
    output_dir: Path = EVALUATION_RESULTS_DIR
    llm: EvaluationLLMConfig = field(default_factory=EvaluationLLMConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


class MedicalLLMGenerator:
    """Shared sync generator for dev-set evaluation."""

    def __init__(self, config: EvaluationLLMConfig):
        self.config = config
        self.llm = ChatOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            **get_qwen_langchain_kwargs(config),
        )

    def generate(self, question: str, contexts: List[str], options: List[str]) -> str:
        prompt = build_medical_eval_prompt(
            question=question,
            options=options,
            context="\n\n".join(f"[{index + 1}] {context}" for index, context in enumerate(contexts)),
        )
        response = self.llm.invoke(prompt)
        return response.content


def load_vector_store(index_path: Path) -> MedicalVectorStore:
    """Load the persisted FAISS store."""
    embeddings = get_langchain_embeddings(
        model_type="huggingface",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = MedicalVectorStore(
        embedding_model=embeddings,
        store_type="faiss",
        persist_directory=str(index_path),
    )
    vectorstore.load(str(index_path))
    return vectorstore


def evaluate_single_question(
    vectorstore: MedicalVectorStore,
    generator: MedicalLLMGenerator,
    item: Dict[str, Any],
    top_k: int,
) -> Dict[str, Any]:
    """Run retrieval and generation for one question."""
    search_results = vectorstore.similarity_search_with_score(item["question"], k=top_k)
    docs = [doc for doc, _ in search_results]
    contexts = [doc.page_content for doc in docs]
    scores = [float(score) for _, score in search_results]
    response = generator.generate(item["question"], contexts, item.get("options", []))
    predicted_answer = extract_answer(response)
    correct_answer = get_correct_answer_letter(item)

    return {
        "question": item["question"],
        "options": item.get("options", []),
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": predicted_answer == correct_answer,
        "response": response,
        "retrieved_docs": len(docs),
        "scores": scores,
        "contexts": contexts,
    }


def evaluate_sync_dataset(
    vectorstore: MedicalVectorStore,
    generator: MedicalLLMGenerator,
    questions: List[Dict[str, Any]],
    top_k: int,
    progress_mgr: Optional[EvaluationProgressManager] = None,
    artifact_paths: Optional[Dict[str, Path]] = None,
    live_config: Optional[Dict[str, Any]] = None,
    extra_sections: Optional[Dict[str, Any]] = None,
    dataset_name: str = "Development Set",
    script_name: str = "complete_eval_dev",
) -> Dict[str, Any]:
    """Evaluate a dataset synchronously."""
    start_time = time.time()
    results: List[Dict[str, Any]] = []
    correct = 0

    for index, item in enumerate(questions, start=1):
        result = evaluate_single_question(vectorstore, generator, item, top_k=top_k)
        results.append(result)
        if result["is_correct"]:
            correct += 1

        if progress_mgr:
            elapsed = time.time() - start_time
            progress_mgr.save_checkpoint(
                dataset_name=dataset_name,
                total_questions=len(questions),
                processed_questions=index,
                current_top_k=top_k,
                results=results,
                correct_count=correct,
                total_count=index,
                elapsed_time=elapsed,
                config={"top_k": top_k},
                script_name=script_name,
            )
            stage_result = progress_mgr.build_stage_result(
                dataset_name=dataset_name,
                total_questions=len(questions),
                processed_questions=index,
                correct_count=correct,
                elapsed_time=elapsed,
                detailed_results=results,
                top_k=top_k,
            )
            progress_mgr.print_progress(
                run_name="NAIVE_RAG",
                dataset_name=dataset_name,
                processed_questions=index,
                total_questions=len(questions),
                correct_count=correct,
                elapsed_time=elapsed,
            )
            if artifact_paths and live_config:
                live_sections = dict(extra_sections or {})
                live_sections["current_stage"] = stage_result
                progress_mgr.write_live_results(
                    artifact_paths=artifact_paths,
                    run_name="NAIVE_RAG",
                    evaluation_type="NAIVE_RAG",
                    config=live_config,
                    stage_result=stage_result,
                    extra_sections=live_sections,
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
    questions: List[Dict[str, Any]],
    config: NaiveRAGEvalConfig,
    top_k: int,
    progress_mgr: Optional[EvaluationProgressManager] = None,
    artifact_paths: Optional[Dict[str, Path]] = None,
    live_config: Optional[Dict[str, Any]] = None,
    extra_sections: Optional[Dict[str, Any]] = None,
    dataset_name: str = "Test Set",
    script_name: str = "complete_eval_test",
) -> Dict[str, Any]:
    """Evaluate a dataset asynchronously."""
    client = create_async_client(config.llm)
    semaphore = asyncio.Semaphore(config.concurrency.max_concurrent)
    rate_limiter = RateLimiter(
        requests_per_second=config.concurrency.requests_per_second,
        burst=config.concurrency.max_concurrent,
    )

    async def evaluate_item(item: Dict[str, Any]) -> Dict[str, Any]:
        search_results = await asyncio.to_thread(
            vectorstore.similarity_search_with_score,
            item["question"],
            top_k,
        )
        docs = [doc for doc, _ in search_results]
        contexts = [doc.page_content for doc in docs]
        scores = [float(score) for _, score in search_results]
        prompt = build_medical_eval_prompt(
            question=item["question"],
            options=item.get("options", []),
            context="\n\n".join(f"[{index + 1}] {context}" for index, context in enumerate(contexts)),
        )

        async with semaphore:
            await rate_limiter.acquire()
            completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **get_qwen_completion_kwargs(config.llm),
            )

        response_content = (
            completion.choices[0].message.content
            or completion.choices[0].message.reasoning_content
            or ""
        )
        predicted_answer = extract_answer(response_content)
        correct_answer = get_correct_answer_letter(item)
        return {
            "question": item["question"],
            "options": item.get("options", []),
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer == correct_answer,
            "response": response_content,
            "retrieved_docs": len(docs),
            "scores": scores,
            "contexts": contexts,
        }

    start_time = time.time()
    results: List[Dict[str, Any]] = []
    correct = 0

    tasks = [evaluate_item(item) for item in questions]
    for index, future in enumerate(asyncio.as_completed(tasks), start=1):
        result = await future
        results.append(result)
        if result["is_correct"]:
            correct += 1

        if progress_mgr:
            elapsed = time.time() - start_time
            progress_mgr.save_checkpoint(
                dataset_name=dataset_name,
                total_questions=len(questions),
                processed_questions=index,
                current_top_k=top_k,
                results=results,
                correct_count=correct,
                total_count=index,
                elapsed_time=elapsed,
                config={"top_k": top_k},
                script_name=script_name,
            )
            stage_result = progress_mgr.build_stage_result(
                dataset_name=dataset_name,
                total_questions=len(questions),
                processed_questions=index,
                correct_count=correct,
                elapsed_time=elapsed,
                detailed_results=results,
                top_k=top_k,
            )
            progress_mgr.print_progress(
                run_name="NAIVE_RAG",
                dataset_name=dataset_name,
                processed_questions=index,
                total_questions=len(questions),
                correct_count=correct,
                elapsed_time=elapsed,
            )
            if artifact_paths and live_config:
                live_sections = dict(extra_sections or {})
                live_sections["current_stage"] = stage_result
                progress_mgr.write_live_results(
                    artifact_paths=artifact_paths,
                    run_name="NAIVE_RAG",
                    evaluation_type="NAIVE_RAG",
                    config=live_config,
                    stage_result=stage_result,
                    extra_sections=live_sections,
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
    generator: MedicalLLMGenerator,
    dev_set: List[Dict[str, Any]],
    config: NaiveRAGEvalConfig,
    progress_mgr: Optional[EvaluationProgressManager] = None,
    artifact_paths: Optional[Dict[str, Path]] = None,
    live_config: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[int, float], Dict[str, Any]]:
    """Search for the best top-k on the dev set."""
    scores: Dict[int, float] = {}
    results_by_k: Dict[int, Dict[str, Any]] = {}
    for k in config.top_k_values:
        result = evaluate_sync_dataset(
            vectorstore=vectorstore,
            generator=generator,
            questions=dev_set,
            top_k=k,
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
            script_name="complete_eval_dev",
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
    """Compute a simple answer-string recall@k metric."""
    recall_scores: Dict[int, float] = {}
    for k in k_values:
        hits = 0
        for item in questions:
            search_results = vectorstore.similarity_search_with_score(item["question"], k=k)
            answer = str(item.get("answer", "")).lower()
            if any(answer and answer in doc.page_content.lower() for doc, _ in search_results):
                hits += 1
        recall_scores[k] = hits / len(questions) if questions else 0.0
    return recall_scores


async def run_complete_evaluation(config: NaiveRAGEvalConfig) -> Dict[str, Any]:
    """Execute the complete naive RAG evaluation flow."""
    questions = load_questions(str(config.question_file))
    dev_set, test_set = split_questions(questions, config.dev_size, config.test_size)

    progress_mgr = EvaluationProgressManager(output_dir=str(config.output_dir))
    artifact_paths = progress_mgr.create_run_artifacts("naive_rag_eval")
    live_config = {
        "dev_set_size": len(dev_set),
        "test_set_size": len(test_set),
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.model,
        "vector_store": str(config.vector_store_path),
        "manual_top_k": config.manual_top_k,
    }
    vectorstore = load_vector_store(config.vector_store_path)
    generator = MedicalLLMGenerator(config.llm)

    if config.manual_top_k is None:
        best_k, dev_scores, dev_results = find_best_top_k(
            vectorstore,
            generator,
            dev_set,
            config,
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

    test_results = await evaluate_async_dataset(
        vectorstore=vectorstore,
        questions=test_set,
        config=config,
        top_k=best_k,
        progress_mgr=progress_mgr,
        artifact_paths=artifact_paths,
        live_config=live_config,
        extra_sections={
            "development_set_evaluation": dev_results,
            "hyperparameter_search": {
                "k_values_tested": config.top_k_values if config.manual_top_k is None else "manual",
                "development_set_accuracy": dev_scores,
                "best_k": best_k,
                "used_manual_top_k": config.manual_top_k is not None,
            },
        },
    )
    recall_scores = calculate_recall_at_k(vectorstore, test_set, [1, 3, 5, 10])
    paths = progress_mgr.write_final_results(
        artifact_paths=artifact_paths,
        run_name="NAIVE_RAG",
        evaluation_type="NAIVE_RAG",
        config=live_config,
        stage_results={
            "development_set_evaluation": dev_results,
            "test_set_evaluation": test_results,
        },
        extra_sections={
            "hyperparameter_search": {
                "k_values_tested": config.top_k_values if config.manual_top_k is None else "manual",
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
