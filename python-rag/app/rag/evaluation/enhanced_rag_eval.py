"""Native enhanced RAG evaluation pipeline."""

from __future__ import annotations

import asyncio
import os
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
    build_eval_result,
    load_questions,
    split_questions,
    update_progress,
)
from .naive_rag_eval import (
    build_query,
    create_llm,
    extract_rag_metadata,
    load_vector_store,
)


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
    use_llm_query_rewrite: bool = True
    use_reranker: bool = True
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-large")
    )
    reranker_device: str = field(
        default_factory=lambda: os.getenv("RAG_RERANKER_DEVICE", "auto")
    )
    llm: EvaluationLLMConfig = field(default_factory=EvaluationLLMConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


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
    initial_elapsed: float = 0.0,
) -> Dict[str, Any]:
    start_time = time.time() - initial_elapsed
    results: List[Dict[str, Any]] = list(initial_results or [])
    correct = initial_correct
    remaining_questions = questions[start_from:]
    semaphore = asyncio.Semaphore(max(1, config.concurrency.max_concurrent))
    rewrite_transform = None

    if config.use_query_rewrite and query_rewriter is not None:
        rewrite_transform = query_rewriter.as_transform(
            use_llm=config.use_llm_query_rewrite
        )

    async def evaluate_item(item: Dict[str, Any]) -> Dict[str, Any]:
        question = item["question"]
        if rewrite_transform is not None:
            rewritten_bundle = await asyncio.to_thread(
                rewrite_transform.run,
                question,
                {"use_llm": config.use_llm_query_rewrite},
            )
            question = rewritten_bundle.query_str

        prompt = build_query(question, item.get("options", []))
        async with semaphore:
            response = await query_engine.aquery(prompt)
        return build_eval_result(item, str(response), extract_rag_metadata(response))

    batch_size = max(1, config.concurrency.max_concurrent)
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
                    extra_sections=None,
                    dataset_name=dataset_name,
                    total_questions=len(questions),
                    processed_questions=processed_questions,
                    correct_count=correct,
                    elapsed=time.time() - start_time,
                    results=results,
                    run_name=run_names.run_name,
                    evaluation_type=run_names.evaluation_type,
                    config_payload={"top_k": config.top_k},
                    script_name=run_names.test_script_name,
                    top_k=config.top_k,
                )

    elapsed = time.time() - start_time
    return {
        "dataset_name": dataset_name,
        "top_k": config.top_k,
        "total_questions": len(questions),
        "processed_questions": len(questions),
        "correct": correct,
        "accuracy": correct / len(questions) if questions else 0.0,
        "elapsed_time": elapsed,
        "questions_per_second": len(questions) / elapsed if elapsed > 0 else 0.0,
        "detailed_results": results,
    }


async def run_enhanced_evaluation(config: EnhancedEvaluationConfig) -> Dict[str, Any]:
    vectorstore = load_vector_store(config.vector_store_path)
    questions = load_questions(str(config.question_file))
    dev_set, test_set = split_questions(questions, config.dev_size, config.test_size)
    run_names = EnhancedEvaluationRunNames()
    query_engine = build_enhanced_query_engine(vectorstore, config)
    query_rewriter = QueryRewritePipeline(
        use_dict=config.use_query_rewrite,
        use_llm=config.use_llm_query_rewrite,
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
        "use_reranker": config.use_reranker,
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
    return {
        "dev_set_size": len(dev_set),
        "test_results": test_results,
        "output_paths": output_paths,
    }