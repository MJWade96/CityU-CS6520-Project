"""Compare two Advanced RAG execution schedules on a 10-question slice.

The script intentionally keeps parameters as module constants to match the
repository convention against command-line arguments. It reuses the existing
Enhanced RAG components and only mirrors the small assembly code needed to
separate retrieval caching from generator execution.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from app.rag.data.data_paths import EVALUATION_RESULTS_DIR, MEDQA_USMLE_DEV_FILE
from app.rag.data.json_utils import save_json_atomic
from app.rag.evaluation.enhanced_rag_eval import (
    EnhancedEvaluationConfig,
    should_use_llm_query_rewrite,
)
from app.rag.evaluation.eval_shared import (
    RateLimiter,
    build_eval_result,
    load_questions,
    split_questions,
)
from app.rag.evaluation.naive_rag_eval import (
    build_query,
    create_llm,
    extract_rag_metadata,
    load_vector_store,
)
from app.rag.retriever.hybrid_retriever import HybridRetriever
from app.rag.retriever.query_rewrite import QueryRewritePipeline
from app.rag.retriever.reranker import RerankerPipeline


TEST_SIZE = 10
DEV_SIZE = 0
OUTPUT_DIR = EVALUATION_RESULTS_DIR
OUTPUT_FILE = OUTPUT_DIR / "advanced_rag_latency_compare.json"


class CachedNodesRetriever(BaseRetriever):
    """Return precomputed final nodes so generation reuses LlamaIndex synthesis."""

    def __init__(self, cached_nodes_by_prompt: Dict[str, List[NodeWithScore]]) -> None:
        super().__init__()
        self.cached_nodes_by_prompt = cached_nodes_by_prompt

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return list(self.cached_nodes_by_prompt[query_bundle.query_str])

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retrieve(query_bundle)


async def close_reused_client(resource: Any) -> None:
    """Close reused OpenAI-compatible clients created by LlamaIndex adapters."""
    for attr_name in ("_aclient", "_client"):
        client = getattr(resource, attr_name, None)
        if client is None:
            continue
        close = getattr(client, "close", None)
        if close is None:
            continue
        result = close()
        if inspect.isawaitable(result):
            await result
        try:
            setattr(resource, attr_name, None)
        except Exception:
            pass


async def close_query_rewriter(query_rewriter: QueryRewritePipeline) -> None:
    """Close the optional LLM rewriter client without changing business modules."""
    llm_rewriter = getattr(query_rewriter, "llm_rewriter", None)
    if llm_rewriter is not None:
        await close_reused_client(llm_rewriter.llm)


async def close_components(components: Dict[str, Any]) -> None:
    """Release clients assembled in this standalone latency script."""
    await close_reused_client(components["llm"])
    for postprocessor in components["node_postprocessors"]:
        session = getattr(postprocessor, "_session", None)
        if session is not None:
            session.close()


@dataclass
class PreparedQuestion:
    index: int
    item: Dict[str, Any]
    prompt: str
    rewritten_question: str
    used_llm_rewrite: bool


@dataclass
class RetrievedQuestion:
    prepared: PreparedQuestion
    nodes: List[NodeWithScore]
    elapsed_seconds: float


def build_query_rewriter(config: EnhancedEvaluationConfig) -> QueryRewritePipeline:
    """Centralize query-rewriter construction to avoid repeating call settings."""
    return QueryRewritePipeline(
        use_dict=config.use_query_rewrite,
        use_llm=config.use_query_rewrite and config.use_llm_query_rewrite,
        llm_provider=config.llm.provider,
        llm_model=config.llm.model,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        llm_temperature=config.llm.temperature,
        llm_enable_thinking=config.llm.enable_thinking,
    )


def build_retrieval_components(vectorstore: Any, config: EnhancedEvaluationConfig) -> Dict[str, Any]:
    """Mirror the existing enhanced engine assembly while exposing each component."""
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

    node_postprocessors = []
    if config.use_reranker:
        reranker = RerankerPipeline(
            use_cross_encoder=True,
            cross_encoder_model=config.reranker_model,
            top_k=reranker_top_k,
            api_url=config.reranker_api_url,
            api_key=config.reranker_api_key,
        )
        if reranker.cross_encoder is not None and reranker.cross_encoder.available:
            node_postprocessors.append(reranker.cross_encoder.model)

    return {
        "llm": llm,
        "retriever": retriever,
        "node_postprocessors": node_postprocessors,
    }


async def prepare_question(
    index: int,
    item: Dict[str, Any],
    query_rewriter: QueryRewritePipeline,
    config: EnhancedEvaluationConfig,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
) -> PreparedQuestion:
    original_question = item.get("question", "")
    used_llm_rewrite = should_use_llm_query_rewrite(
        original_question,
        query_rewriter,
        config,
    )
    rewritten_question = original_question
    if config.use_query_rewrite:
        rewritten_question, _ = await query_rewriter.arewrite(
            rewritten_question,
            rate_limiter=rate_limiter,
            api_semaphore=semaphore,
            use_llm=used_llm_rewrite,
        )

    return PreparedQuestion(
        index=index,
        item=item,
        prompt=build_query(rewritten_question, item.get("options", [])),
        rewritten_question=rewritten_question,
        used_llm_rewrite=used_llm_rewrite,
    )


async def retrieve_final_nodes(
    prepared: PreparedQuestion,
    retriever: Any,
    node_postprocessors: List[Any],
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
) -> RetrievedQuestion:
    started_at = time.perf_counter()
    query_bundle = QueryBundle(prepared.prompt)
    async with semaphore:
        await rate_limiter.acquire()
        nodes = await retriever.aretrieve(query_bundle)
        for postprocessor in node_postprocessors:
            nodes = await postprocessor.apostprocess_nodes(
                nodes,
                query_bundle=query_bundle,
            )
    return RetrievedQuestion(
        prepared=prepared,
        nodes=nodes,
        elapsed_seconds=time.perf_counter() - started_at,
    )


async def run_retrieve_first(
    questions: List[Dict[str, Any]],
    vectorstore: Any,
    config: EnhancedEvaluationConfig,
) -> Dict[str, Any]:
    components = build_retrieval_components(vectorstore, config)
    query_rewriter = build_query_rewriter(config)
    batch_size = max(1, config.concurrency.max_concurrent)
    semaphore = asyncio.Semaphore(batch_size)
    rate_limiter = RateLimiter(
        requests_per_second=config.concurrency.requests_per_second,
        burst=batch_size,
    )

    try:
        started_at = time.perf_counter()
        prepared_questions = await asyncio.gather(
            *(
                prepare_question(
                    index,
                    item,
                    query_rewriter,
                    config,
                    rate_limiter,
                    semaphore,
                )
                for index, item in enumerate(questions, start=1)
            )
        )
        rewrite_elapsed = time.perf_counter() - started_at

        retrieval_started_at = time.perf_counter()
        retrieved_questions = await asyncio.gather(
            *(
                retrieve_final_nodes(
                    prepared,
                    components["retriever"],
                    components["node_postprocessors"],
                    semaphore,
                    rate_limiter,
                )
                for prepared in prepared_questions
            )
        )
        retrieval_elapsed = time.perf_counter() - retrieval_started_at

        cached_nodes_by_prompt = {
            retrieved.prepared.prompt: retrieved.nodes for retrieved in retrieved_questions
        }
        cached_query_engine = RetrieverQueryEngine.from_args(
            retriever=CachedNodesRetriever(cached_nodes_by_prompt),
            llm=components["llm"],
            use_async=True,
        )

        generation_started_at = time.perf_counter()

        async def generate(retrieved: RetrievedQuestion) -> Dict[str, Any]:
            async with semaphore:
                await rate_limiter.acquire()
                started = time.perf_counter()
                response = await cached_query_engine.aquery(retrieved.prepared.prompt)
            result = build_eval_result(
                retrieved.prepared.item,
                str(response),
                extract_rag_metadata(response),
            )
            result["latency_seconds"] = time.perf_counter() - started
            result["retrieval_latency_seconds"] = retrieved.elapsed_seconds
            result["used_llm_rewrite"] = retrieved.prepared.used_llm_rewrite
            result["rewritten_question"] = retrieved.prepared.rewritten_question
            return result

        detailed_results = await asyncio.gather(
            *(generate(item) for item in retrieved_questions)
        )
        generation_elapsed = time.perf_counter() - generation_started_at
        total_elapsed = time.perf_counter() - started_at
        correct = sum(1 for result in detailed_results if result["is_correct"])

        return {
            "name": "retrieve_first_then_batch_generate",
            "total_questions": len(questions),
            "correct": correct,
            "accuracy": correct / len(questions) if questions else 0.0,
            "total_elapsed_seconds": total_elapsed,
            "average_seconds_per_question": (
                total_elapsed / len(questions) if questions else 0.0
            ),
            "rewrite_elapsed_seconds": rewrite_elapsed,
            "retrieval_elapsed_seconds": retrieval_elapsed,
            "generation_elapsed_seconds": generation_elapsed,
            "detailed_results": detailed_results,
        }
    finally:
        await close_query_rewriter(query_rewriter)
        await close_components(components)


async def run_full_rag_batch(
    questions: List[Dict[str, Any]],
    vectorstore: Any,
    config: EnhancedEvaluationConfig,
) -> Dict[str, Any]:
    components = build_retrieval_components(vectorstore, config)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=components["retriever"],
        llm=components["llm"],
        node_postprocessors=components["node_postprocessors"],
        use_async=True,
    )
    query_rewriter = build_query_rewriter(config)
    batch_size = max(1, config.concurrency.max_concurrent)
    semaphore = asyncio.Semaphore(batch_size)
    rate_limiter = RateLimiter(
        requests_per_second=config.concurrency.requests_per_second,
        burst=batch_size,
    )

    async def evaluate(index: int, item: Dict[str, Any]) -> Dict[str, Any]:
        prepared = await prepare_question(
            index,
            item,
            query_rewriter,
            config,
            rate_limiter,
            semaphore,
        )
        async with semaphore:
            await rate_limiter.acquire()
            started = time.perf_counter()
            response = await query_engine.aquery(prepared.prompt)
        result = build_eval_result(item, str(response), extract_rag_metadata(response))
        result["latency_seconds"] = time.perf_counter() - started
        result["used_llm_rewrite"] = prepared.used_llm_rewrite
        result["rewritten_question"] = prepared.rewritten_question
        return result

    try:
        started_at = time.perf_counter()
        detailed_results = await asyncio.gather(
            *(evaluate(index, item) for index, item in enumerate(questions, start=1))
        )
        total_elapsed = time.perf_counter() - started_at
        correct = sum(1 for result in detailed_results if result["is_correct"])

        return {
            "name": "direct_batch_full_rag",
            "total_questions": len(questions),
            "correct": correct,
            "accuracy": correct / len(questions) if questions else 0.0,
            "total_elapsed_seconds": total_elapsed,
            "average_seconds_per_question": (
                total_elapsed / len(questions) if questions else 0.0
            ),
            "detailed_results": detailed_results,
        }
    finally:
        await close_query_rewriter(query_rewriter)
        await close_components(components)


def build_report(
    config: EnhancedEvaluationConfig,
    setup_elapsed: float,
    retrieve_first: Dict[str, Any],
    full_rag: Dict[str, Any],
) -> Dict[str, Any]:
    """Keep output serialization in one place for reproducible comparison runs."""
    return {
        "experiment": "advanced_rag_latency_compare",
        "question_count": TEST_SIZE,
        "timing_scope": "execution_only_excludes_pipeline_initialization",
        "setup_elapsed_seconds": setup_elapsed,
        "config": {
            "top_k": config.top_k,
            "retrieval_top_k": config.resolved_retrieval_top_k,
            "reranker_top_k": config.resolved_reranker_top_k,
            "hybrid_alpha": config.hybrid_alpha,
            "use_hybrid_retrieval": config.use_hybrid_retrieval,
            "use_query_rewrite": config.use_query_rewrite,
            "use_llm_query_rewrite": config.use_llm_query_rewrite,
            "llm_query_rewrite_mode": config.llm_query_rewrite_mode,
            "use_reranker": config.use_reranker,
            "max_concurrent": config.concurrency.max_concurrent,
            "rpm_limit": config.concurrency.rpm_limit,
        },
        "results": {
            retrieve_first["name"]: retrieve_first,
            full_rag["name"]: full_rag,
        },
        "summary": {
            "retrieve_first_total_seconds": retrieve_first["total_elapsed_seconds"],
            "retrieve_first_average_seconds_per_question": retrieve_first[
                "average_seconds_per_question"
            ],
            "full_rag_total_seconds": full_rag["total_elapsed_seconds"],
            "full_rag_average_seconds_per_question": full_rag[
                "average_seconds_per_question"
            ],
            "retrieve_first_minus_full_rag_seconds": (
                retrieve_first["total_elapsed_seconds"]
                - full_rag["total_elapsed_seconds"]
            ),
        },
    }


async def main_async() -> None:
    config = EnhancedEvaluationConfig(
        dev_size=DEV_SIZE,
        test_size=TEST_SIZE,
        question_file=MEDQA_USMLE_DEV_FILE,
    )

    print("Initializing Advanced RAG components...")
    setup_started_at = time.perf_counter()
    vectorstore = load_vector_store(config.vector_store_path)
    try:
        questions = load_questions(str(config.question_file))
        _, test_set = split_questions(questions, config.dev_size, config.test_size)
        setup_elapsed = time.perf_counter() - setup_started_at

        print(f"Loaded {len(test_set)} questions for timing.")
        print("Running retrieve-first schedule...")
        retrieve_first = await run_retrieve_first(test_set, vectorstore, config)
        print(
            "  retrieve-first: "
            f"{retrieve_first['total_elapsed_seconds']:.2f}s total, "
            f"{retrieve_first['average_seconds_per_question']:.2f}s/question"
        )

        print("Running direct full-RAG batch schedule...")
        full_rag = await run_full_rag_batch(test_set, vectorstore, config)
        print(
            "  full-RAG batch: "
            f"{full_rag['total_elapsed_seconds']:.2f}s total, "
            f"{full_rag['average_seconds_per_question']:.2f}s/question"
        )

        report = build_report(config, setup_elapsed, retrieve_first, full_rag)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_json_atomic(OUTPUT_FILE, report, indent=2)

        print("Summary:")
        print(json.dumps(report["summary"], indent=2))
        print(f"Results JSON: {OUTPUT_FILE}")
    finally:
        await close_reused_client(vectorstore._embed_model)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
