"""
Naive RAG evaluation pipeline.

The core evaluation logic lives here so ``complete_eval.py`` can stay as a thin
entrypoint and other scripts can reuse the same retrieval/generation flow.
"""

from __future__ import annotations

import asyncio
import json
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
            progress_mgr.save_checkpoint(
                dataset_name=dataset_name,
                total_questions=len(questions),
                processed_questions=index,
                current_top_k=top_k,
                results=results,
                correct_count=correct,
                total_count=index,
                elapsed_time=time.time() - start_time,
                config={"top_k": top_k},
                script_name=script_name,
            )

    elapsed = time.time() - start_time
    return {
        "dataset_name": dataset_name,
        "top_k": top_k,
        "total_questions": len(questions),
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
            progress_mgr.save_checkpoint(
                dataset_name=dataset_name,
                total_questions=len(questions),
                processed_questions=index,
                current_top_k=top_k,
                results=results,
                correct_count=correct,
                total_count=index,
                elapsed_time=time.time() - start_time,
                config={"top_k": top_k},
                script_name=script_name,
            )

    elapsed = time.time() - start_time
    return {
        "dataset_name": dataset_name,
        "top_k": top_k,
        "total_questions": len(questions),
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
) -> Tuple[int, Dict[int, float]]:
    """Search for the best top-k on the dev set."""
    scores: Dict[int, float] = {}
    for k in config.top_k_values:
        result = evaluate_sync_dataset(
            vectorstore=vectorstore,
            generator=generator,
            questions=dev_set,
            top_k=k,
            progress_mgr=progress_mgr,
            dataset_name=f"Development Set (k={k})",
            script_name="complete_eval_dev",
        )
        scores[k] = result["accuracy"]
    best_k = max(scores, key=scores.get)
    return best_k, scores


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


def save_results(
    config: NaiveRAGEvalConfig,
    best_k: int,
    dev_scores: Dict[int, float],
    dev_size: int,
    test_size: int,
    test_results: Dict[str, Any],
    recall_scores: Dict[int, float],
) -> Dict[str, Path]:
    """Persist JSON and text summaries."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = config.output_dir / f"complete_eval_{timestamp}.json"
    summary_path = config.output_dir / f"evaluation_summary_{timestamp}.txt"

    payload = {
        "config": {
            "dev_set_size": dev_size,
            "test_set_size": test_size,
            "llm_provider": config.llm.provider,
            "llm_model": config.llm.model,
            "vector_store": str(config.vector_store_path),
            "manual_top_k": config.manual_top_k,
        },
        "hyperparameter_search": {
            "k_values_tested": config.top_k_values if config.manual_top_k is None else "manual",
            "development_set_accuracy": dev_scores,
            "best_k": best_k,
            "used_manual_top_k": config.manual_top_k is not None,
        },
        "test_set_evaluation": test_results,
        "retrieval_recall_at_k": recall_scores,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Medical RAG System - Naive RAG Evaluation Summary\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"LLM: {config.llm.provider}/{config.llm.model}\n")
        handle.write(f"Vector Store: {config.vector_store_path}\n\n")
        handle.write("Dataset Split:\n")
        handle.write(f"  Development Set: {dev_size} questions\n")
        handle.write(f"  Test Set: {test_size} questions\n\n")
        if config.manual_top_k is None:
            handle.write("Hyperparameter Search (Development Set):\n")
            for k, score in dev_scores.items():
                handle.write(f"  top-k={k}: Accuracy = {score:.4f}\n")
        else:
            handle.write(f"Manual top-k: {best_k}\n")
        handle.write(f"\nTest Accuracy: {test_results['accuracy']:.4f}\n")
        handle.write(f"Test Correct: {test_results['correct']}/{test_results['total_questions']}\n")
        handle.write(f"Time: {test_results['elapsed_time']:.1f}s\n\n")
        handle.write("Retrieval Recall@k:\n")
        for k, score in recall_scores.items():
            handle.write(f"  R@{k}: {score:.4f}\n")

    return {"json": json_path, "summary": summary_path}


async def run_complete_evaluation(config: NaiveRAGEvalConfig) -> Dict[str, Any]:
    """Execute the complete naive RAG evaluation flow."""
    questions = load_questions(str(config.question_file))
    dev_set, test_set = split_questions(questions, config.dev_size, config.test_size)

    progress_mgr = EvaluationProgressManager(output_dir=str(config.output_dir))
    vectorstore = load_vector_store(config.vector_store_path)
    generator = MedicalLLMGenerator(config.llm)

    if config.manual_top_k is None:
        best_k, dev_scores = find_best_top_k(vectorstore, generator, dev_set, config, progress_mgr)
    else:
        best_k, dev_scores = config.manual_top_k, {}

    test_results = await evaluate_async_dataset(
        vectorstore=vectorstore,
        questions=test_set,
        config=config,
        top_k=best_k,
        progress_mgr=progress_mgr,
    )
    recall_scores = calculate_recall_at_k(vectorstore, test_set, [1, 3, 5, 10])
    paths = save_results(
        config=config,
        best_k=best_k,
        dev_scores=dev_scores,
        dev_size=len(dev_set),
        test_size=len(test_set),
        test_results=test_results,
        recall_scores=recall_scores,
    )

    return {
        "best_k": best_k,
        "dev_scores": dev_scores,
        "test_results": test_results,
        "recall_scores": recall_scores,
        "output_paths": paths,
    }
