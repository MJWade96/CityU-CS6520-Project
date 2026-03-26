"""
Enhanced Medical RAG Evaluation System

Integrates all Phase 1 and Phase 2 optimizations:
- Phase 1:
  * Hybrid retrieval (Dense + BM25 with RRF fusion)
  * Query rewriting (dictionary + LLM-based)
  * Prompt optimization (Chain-of-Thought, structured output)

- Phase 2:
  * Semantic chunking with sliding window
  * Parent-Child chunk association
  * Metadata enhancement
  * Cross-Encoder reranking

Usage:
    python enhanced_eval.py
"""

import asyncio
import hashlib
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# Import optimization modules
from app.rag.hybrid_retriever import HybridRetriever, AdaptiveRetriever
from app.rag.query_rewrite import QueryRewritePipeline
from app.rag.reranker import RerankerPipeline
from app.rag.chunking import SemanticChunker, ParentChildChunker
from app.rag.metadata_enhancement import MetadataGenerator, RuleBasedMetadataGenerator
from app.rag.progress_manager import EvaluationProgressManager
from app.rag.data_paths import EVALUATION_DIR, EVALUATION_RESULTS_DIR, FAISS_INDEX_DIR
from app.rag.eval_shared import (
    ConcurrencyConfig,
    EvaluationLLMConfig,
    RateLimiter,
    build_medical_eval_prompt,
    extract_answer,
    get_qwen_langchain_kwargs,
    parse_optional_bool_env,
)


# ============================================================
# Configuration
# ============================================================


class EnhancedEvaluationConfig:
    """Enhanced evaluation configuration"""

    # Dataset split
    # Match the legacy no-RAG benchmark: dev uses questions[0:50],
    # test uses questions[50:100].
    DEV_SET_SIZE = 50
    TEST_SET_SIZE = 50

    # LLM Configuration (联通云 DeepSeek V3.2)
    LLM_PROVIDER = "Qwen3-4B"
    LLM_MODEL = "8606056bfe0c49448d92587452d1f2fc"
    LLM_TEMPERATURE = 0.1
    LLM_BASE_URL = "https://wishub-x6.ctyun.cn/v1"
    LLM_API_KEY = "4dbe3bec3ee548d28b649b324e741939"
    QUERY_REWRITE_PROVIDER = os.getenv("RAG_QUERY_REWRITE_PROVIDER", LLM_PROVIDER)
    QUERY_REWRITE_MODEL = os.getenv("RAG_QUERY_REWRITE_MODEL", LLM_MODEL)
    QUERY_REWRITE_TEMPERATURE = float(
        os.getenv("RAG_QUERY_REWRITE_TEMPERATURE", str(LLM_TEMPERATURE))
    )
    QUERY_REWRITE_MAX_TOKENS = int(os.getenv("RAG_QUERY_REWRITE_MAX_TOKENS", "200"))
    QUERY_REWRITE_BASE_URL = os.getenv("RAG_QUERY_REWRITE_BASE_URL", LLM_BASE_URL)
    QUERY_REWRITE_API_KEY = os.getenv("RAG_QUERY_REWRITE_API_KEY", LLM_API_KEY)
    QUERY_REWRITE_ENABLE_THINKING = parse_optional_bool_env(
        "RAG_QUERY_REWRITE_ENABLE_THINKING",
        default=False,
    )

    # Retrieval configuration
    TOP_K_VALUES = [1, 3, 5, 10]
    DEFAULT_TOP_K = 5

    # Optimization flags
    USE_HYBRID_RETRIEVAL = True
    USE_QUERY_REWRITE = True
    USE_RERANKER = True
    USE_COT_PROMPT = False
    USE_ADAPTIVE_RETRIEVAL = False
    CONCURRENCY = ConcurrencyConfig()

    # File paths
    VECTOR_STORE_PATH = str(FAISS_INDEX_DIR)
    QUESTION_FILE = str(EVALUATION_DIR / "medqa.json")
    OUTPUT_DIR = str(EVALUATION_RESULTS_DIR)
    CACHE_DIR = str(EVALUATION_RESULTS_DIR / "cache")


# ============================================================
# Enhanced LLM Generator
# ============================================================


class EnhancedMedicalLLMGenerator:
    """Enhanced LLM Generator with optimized prompts"""

    def __init__(
        self,
        provider: str = "Qwen3-4B",
        model: str = "8606056bfe0c49448d92587452d1f2fc",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the shared-eval-style LLM generator."""
        self.provider = provider
        self.model = model
        self.temperature = temperature
        # Get API credentials
        self.api_key = api_key or "4dbe3bec3ee548d28b649b324e741939"
        self.base_url = base_url or "https://wishub-x6.ctyun.cn/v1"

        llm_config = EvaluationLLMConfig(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **get_qwen_langchain_kwargs(llm_config),
        )

    def generate(
        self,
        question: str,
        contexts: List[str],
        options: Optional[List[str]] = None,
    ) -> str:
        """Generate an answer using the shared evaluation prompt."""
        prompt = build_medical_eval_prompt(
            question=question,
            options=options or [],
            context="\n\n".join(
                f"[{index + 1}] {context}"
                for index, context in enumerate(contexts)
            ),
        )

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract the final answer using the shared helper."""
        return extract_answer(response)


# ============================================================
# Enhanced RAG Pipeline
# ============================================================


class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline with all optimizations

    Features:
    - Hybrid retrieval (Dense + BM25)
    - Query rewriting
    - Reranking
    - Adaptive retrieval
    """

    def __init__(
        self,
        embedding_model,
        documents: List[Document],
        config: EnhancedEvaluationConfig,
        vectorstore=None,
        bm25_cache_path: Optional[str] = None,
    ):
        """Initialize enhanced RAG pipeline"""
        self.config = config
        self.documents = documents
        self.embedding_model = embedding_model
        self.vectorstore = vectorstore
        self.bm25_cache_path = bm25_cache_path

        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            embedding_model=embedding_model,
            documents=documents,
            dense_weight=0.5,
            dense_vectorstore=vectorstore,
            bm25_cache_path=bm25_cache_path,
        )

        # Initialize adaptive retriever
        self.adaptive_retriever = AdaptiveRetriever(self.hybrid_retriever)

        # Initialize query rewrite pipeline
        self.query_rewriter = QueryRewritePipeline(
            use_dict=True,
            use_llm=config.USE_QUERY_REWRITE,
            use_expansion=False,
            llm_provider=config.QUERY_REWRITE_PROVIDER,
            llm_model=config.QUERY_REWRITE_MODEL,
            api_key=config.QUERY_REWRITE_API_KEY,
            base_url=config.QUERY_REWRITE_BASE_URL,
            llm_temperature=config.QUERY_REWRITE_TEMPERATURE,
            llm_max_tokens=config.QUERY_REWRITE_MAX_TOKENS,
            llm_enable_thinking=config.QUERY_REWRITE_ENABLE_THINKING,
        )

        # Initialize reranker
        self.reranker = RerankerPipeline(
            use_cross_encoder=config.USE_RERANKER,
            use_mmr=False,
            use_lost_in_middle=False,
            top_k=config.DEFAULT_TOP_K,
        )

        # Initialize LLM generator
        self.llm_generator = EnhancedMedicalLLMGenerator(
            provider=config.LLM_PROVIDER,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_rewrite: bool = True,
        use_rerank: bool = True,
        use_adaptive: bool = False,
    ) -> List[Tuple[Document, float]]:
        """
        Enhanced retrieval with query rewriting and reranking

        Args:
            query: Original query
            top_k: Number of documents to return
            use_rewrite: Use query rewriting
            use_rerank: Use reranking
            use_adaptive: Use adaptive retrieval

        Returns:
            List of (document, score) tuples
        """
        # Step 1: Query rewriting
        if use_rewrite:
            primary_query, all_queries = self.query_rewriter.rewrite(
                query, mode="single"
            )
        else:
            primary_query = query

        # Step 2: Retrieval
        if use_adaptive:
            results = self.adaptive_retriever.search(primary_query, k=top_k * 2)
        else:
            results = self.hybrid_retriever.search(
                primary_query, k=top_k * 2, use_hybrid=self.config.USE_HYBRID_RETRIEVAL
            )

        # Step 3: Reranking
        if use_rerank and self.reranker:
            results = self.reranker.rerank(primary_query, results)

        # Return top-k
        return results[:top_k]

    def answer(
        self,
        query: str,
        options: List[str] = None,
        top_k: int = 5,
        use_rewrite: bool = True,
        use_rerank: bool = True,
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve and answer

        Args:
            query: Query string
            options: Answer options
            top_k: Number of documents to retrieve
            use_rewrite: Use query rewriting
            use_rerank: Use reranking

        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve
        results = self.retrieve(
            query, top_k=top_k, use_rewrite=use_rewrite, use_rerank=use_rerank
        )

        # Extract contexts
        contexts = [doc.page_content for doc, score in results]

        # Generate answer
        response = self.llm_generator.generate(query, contexts, options)
        predicted_answer = self.llm_generator.extract_answer(response)

        return {
            "query": query,
            "retrieved_docs": results,
            "contexts": contexts,
            "response": response,
            "predicted_answer": predicted_answer,
        }

# ============================================================
# Evaluation Functions
# ============================================================


def load_questions(question_file: str) -> List[Dict]:
    """Load MedQA questions"""
    print(f"Loading questions from {question_file}...")

    if not os.path.exists(question_file):
        print(f"ERROR: File not found: {question_file}")
        return []

    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"[OK] Loaded {len(questions)} questions")
    return questions


def load_vector_store(config: EnhancedEvaluationConfig):
    """Load vector store with documents"""
    print(f"\nLoading vector store from {config.VECTOR_STORE_PATH}...")

    if not os.path.exists(config.VECTOR_STORE_PATH):
        print(f"ERROR: Vector store not found: {config.VECTOR_STORE_PATH}")
        print("Run build_vector_index.py first")
        return None, None, None

    # Load embeddings
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Load vector store using MedicalVectorStore
    from app.rag.vector_store import MedicalVectorStore

    vectorstore = MedicalVectorStore(
        embedding_model=embeddings,
        store_type="faiss",
        persist_directory=config.VECTOR_STORE_PATH,
    )
    vectorstore.load(config.VECTOR_STORE_PATH)

    print("[OK] Vector store loaded")

    # Extract documents from vectorstore
    if hasattr(vectorstore, "documents"):
        documents = vectorstore.documents
    else:
        # Fallback: create empty list
        documents = []

    return embeddings, documents, vectorstore


def get_compatible_resume_info(
    progress_mgr: EvaluationProgressManager,
    script_name: str,
    expected_total_questions: int,
) -> Optional[Dict[str, Any]]:
    """Return resume info only when the checkpoint matches the current dataset size."""
    checkpoint = progress_mgr.load_checkpoint(script_name)
    if not checkpoint:
        return None

    if checkpoint.total_questions != expected_total_questions:
        print(
            f"[resume][{script_name}] ignoring stale checkpoint "
            f"({checkpoint.total_questions} questions) for current dataset "
            f"({expected_total_questions} questions)"
        )
        progress_mgr.clear_checkpoint(script_name)
        return None

    if checkpoint.processed_questions >= checkpoint.total_questions:
        progress_mgr.clear_checkpoint(script_name)
        return None

    return {
        "start_from": checkpoint.processed_questions,
        "results": checkpoint.results,
        "correct_count": checkpoint.correct_count,
        "total_count": checkpoint.total_count,
        "elapsed_time": checkpoint.elapsed_time,
        "current_top_k": checkpoint.current_top_k,
        "config": checkpoint.config,
    }


def build_bm25_cache_path(
    config: EnhancedEvaluationConfig,
    documents: List[Document],
) -> Path:
    """Build a stable cache path for the BM25 retriever index."""
    cache_dir = Path(config.CACHE_DIR)
    build_metadata_path = Path(config.VECTOR_STORE_PATH) / "build_metadata.json"
    metadata_fingerprint = ""
    if build_metadata_path.exists():
        metadata_fingerprint = build_metadata_path.read_text(encoding="utf-8")

    signature_payload = {
        "vector_store_path": str(Path(config.VECTOR_STORE_PATH).resolve()),
        "document_count": len(documents),
        "metadata_fingerprint": metadata_fingerprint,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
    }
    digest = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return cache_dir / f"enhanced_bm25_{digest}.pkl"


def build_progress_config(
    pipeline: EnhancedRAGPipeline,
    top_k: int,
) -> Dict[str, Any]:
    """Build the checkpoint/live-results config payload."""
    reranker_stats = pipeline.reranker.get_stats() if pipeline.reranker else {}
    return {
        "top_k": top_k,
        "llm_provider": pipeline.config.LLM_PROVIDER,
        "llm_model": pipeline.config.LLM_MODEL,
        "query_rewrite_provider": pipeline.config.QUERY_REWRITE_PROVIDER,
        "query_rewrite_model": pipeline.config.QUERY_REWRITE_MODEL,
        "query_rewrite_temperature": pipeline.config.QUERY_REWRITE_TEMPERATURE,
        "query_rewrite_max_tokens": pipeline.config.QUERY_REWRITE_MAX_TOKENS,
        "query_rewrite_enable_thinking": pipeline.config.QUERY_REWRITE_ENABLE_THINKING,
        "hybrid_retrieval": pipeline.config.USE_HYBRID_RETRIEVAL,
        "query_rewrite": pipeline.config.USE_QUERY_REWRITE,
        "reranker": pipeline.config.USE_RERANKER,
        "reranker_available": reranker_stats.get("cross_encoder_available", False),
        "max_concurrent": pipeline.config.CONCURRENCY.max_concurrent,
        "rpm_limit": pipeline.config.CONCURRENCY.rpm_limit,
    }


async def evaluate_with_pipeline_async(
    pipeline: EnhancedRAGPipeline,
    questions: List[Dict],
    top_k: int = 5,
    dataset_name: str = "Dataset",
    progress_mgr: Optional[EvaluationProgressManager] = None,
    start_from: int = 0,
    initial_results: Optional[List[Dict]] = None,
    initial_correct: int = 0,
    initial_total: int = 0,
    initial_elapsed: float = 0.0,
    script_name: Optional[str] = None,
    artifact_paths: Optional[Dict[str, Path]] = None,
    live_config: Optional[Dict[str, Any]] = None,
    extra_sections: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate using the enhanced pipeline with batch concurrency."""
    print(f"\n{'=' * 60}")
    if start_from > 0:
        print(f"Resuming {dataset_name} (top-k={top_k}) from question {start_from + 1}")
    else:
        print(f"Evaluating {dataset_name} (top-k={top_k})")
    print(f"{'=' * 60}")

    start_time = time.time() - initial_elapsed
    results = list(initial_results or [])
    correct = initial_correct
    total = initial_total

    questions_to_process = questions[start_from:]
    progress_config = build_progress_config(pipeline, top_k=top_k)
    batch_size = max(1, pipeline.config.CONCURRENCY.max_concurrent)
    rate_limiter = RateLimiter(
        requests_per_second=pipeline.config.CONCURRENCY.requests_per_second,
        burst=batch_size,
    )

    async def evaluate_item(q: Dict[str, Any]) -> Dict[str, Any]:
        question_text = q.get("question", "")
        options = q.get("options", [])
        correct_answer = q.get("answer", "")
        answer_index = q.get("answer_index", -1)

        if answer_index >= 0:
            correct_answer_letter = chr(65 + answer_index)
        else:
            correct_answer_letter = correct_answer

        try:
            await rate_limiter.acquire()
            result = await asyncio.to_thread(
                pipeline.answer,
                question_text,
                options,
                top_k,
                pipeline.config.USE_QUERY_REWRITE,
                pipeline.config.USE_RERANKER,
            )

            is_correct = result["predicted_answer"] == correct_answer_letter.upper()

            evaluation_result = {
                "question": question_text,
                "options": options,
                "correct_answer": correct_answer_letter,
                "predicted_answer": result["predicted_answer"],
                "is_correct": is_correct,
                "response": result["response"],
                "retrieved_docs": len(result["retrieved_docs"]),
                "error": None,
            }
            if str(result["response"]).startswith("Error generating answer:"):
                evaluation_result["error"] = result["response"]
        except Exception as e:
            evaluation_result = {
                "question": question_text,
                "options": options,
                "correct_answer": correct_answer_letter,
                "predicted_answer": None,
                "is_correct": False,
                "response": f"Error generating answer: {str(e)}",
                "retrieved_docs": 0,
                "error": str(e),
            }

        return evaluation_result

    for batch_start in range(0, len(questions_to_process), batch_size):
        batch = questions_to_process[batch_start : batch_start + batch_size]
        batch_results = await asyncio.gather(*(evaluate_item(item) for item in batch))

        for offset, evaluation_result in enumerate(batch_results, start=1):
            processed_questions = start_from + batch_start + offset

            if evaluation_result["is_correct"]:
                correct += 1
            total += 1
            results.append(evaluation_result)

            elapsed = time.time() - start_time

            if evaluation_result.get("error"):
                print(f"  ERROR on question {processed_questions}: {evaluation_result['error']}")

            if progress_mgr:
                progress_mgr.print_progress(
                    run_name="ENHANCED_RAG",
                    dataset_name=dataset_name,
                    processed_questions=processed_questions,
                    total_questions=len(questions),
                    correct_count=correct,
                    elapsed_time=elapsed,
                )

                progress_mgr.save_checkpoint(
                    dataset_name=dataset_name,
                    total_questions=len(questions),
                    processed_questions=processed_questions,
                    current_top_k=top_k,
                    results=results,
                    correct_count=correct,
                    total_count=total,
                    elapsed_time=elapsed,
                    config=progress_config,
                    script_name=script_name or "enhanced_eval",
                    error_message=evaluation_result.get("error"),
                )
                if artifact_paths and live_config:
                    stage_result = progress_mgr.build_stage_result(
                        dataset_name=dataset_name,
                        total_questions=len(questions),
                        processed_questions=processed_questions,
                        correct_count=correct,
                        elapsed_time=elapsed,
                        detailed_results=results,
                        top_k=top_k,
                    )
                    live_sections = dict(extra_sections or {})
                    live_sections["current_stage"] = stage_result
                    progress_mgr.write_live_results(
                        artifact_paths=artifact_paths,
                        run_name="ENHANCED_RAG",
                        evaluation_type="ENHANCED_RAG",
                        config=live_config,
                        stage_result=stage_result,
                        extra_sections=live_sections,
                    )

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0

    return {
        "dataset_name": dataset_name,
        "total_questions": total,
        "processed_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_time": elapsed,
        "questions_per_second": total / elapsed if elapsed > 0 else 0,
        "top_k": top_k,
        "detailed_results": results,
    }


# ============================================================
# Main Evaluation Pipeline
# ============================================================


async def main_async():
    """Main evaluation function with checkpoint support."""
    print("=" * 60)
    print("Enhanced Medical RAG System - Complete Evaluation")
    print("Phase 1 + Phase 2 Optimizations")
    print("=" * 60)

    # Load configuration
    config = EnhancedEvaluationConfig()

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Initialize progress manager
    progress_mgr = EvaluationProgressManager(output_dir=config.OUTPUT_DIR)
    artifact_paths = progress_mgr.create_run_artifacts("enhanced_rag_eval")

    # Load questions
    questions = load_questions(config.QUESTION_FILE)

    if not questions:
        print("\nNo questions loaded. Exiting...")
        return

    # Evaluate only the legacy test slice questions[50:100].
    test_start_index = config.DEV_SET_SIZE
    test_set = questions[test_start_index : test_start_index + config.TEST_SET_SIZE]

    print(f"\nEvaluation Scope:")
    print(f"  Only evaluating test set")
    print(f"  Test set: {len(test_set)} questions")
    print(
        f"  Test question range: "
        f"[{test_start_index}:{test_start_index + len(test_set)}]"
    )

    if not test_set:
        print("\nNo test questions available. Exiting...")
        return

    # Load vector store
    embeddings, documents, vectorstore = load_vector_store(config)

    if embeddings is None or documents is None or vectorstore is None:
        print("\nFailed to load vector store. Exiting...")
        return

    bm25_cache_path = build_bm25_cache_path(config, documents)

    # Initialize enhanced pipeline
    print("\nInitializing Enhanced RAG Pipeline...")
    print(f"  Hybrid Retrieval: {config.USE_HYBRID_RETRIEVAL}")
    print(f"  Query Rewrite: {config.USE_QUERY_REWRITE}")
    print(f"  Reranker: {config.USE_RERANKER}")
    print(f"  CoT Prompting: {config.USE_COT_PROMPT}")
    print(f"  Adaptive Retrieval: {config.USE_ADAPTIVE_RETRIEVAL}")
    print(f"  Max Concurrent: {config.CONCURRENCY.max_concurrent}")
    if config.USE_QUERY_REWRITE:
        print(
            f"  Query Rewrite Model: {config.QUERY_REWRITE_MODEL} "
            f"(temp={config.QUERY_REWRITE_TEMPERATURE}, "
            f"max_tokens={config.QUERY_REWRITE_MAX_TOKENS}, "
            f"thinking={config.QUERY_REWRITE_ENABLE_THINKING})"
        )

    pipeline = EnhancedRAGPipeline(
        embedding_model=embeddings,
        documents=documents,
        config=config,
        vectorstore=vectorstore,
        bm25_cache_path=str(bm25_cache_path),
    )

    live_config = {
        "test_set_size": len(test_set),
        "test_question_start_index": test_start_index,
        "test_question_end_index": test_start_index + len(test_set) - 1,
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": config.LLM_MODEL,
        "query_rewrite_provider": config.QUERY_REWRITE_PROVIDER,
        "query_rewrite_model": config.QUERY_REWRITE_MODEL,
        "query_rewrite_temperature": config.QUERY_REWRITE_TEMPERATURE,
        "query_rewrite_max_tokens": config.QUERY_REWRITE_MAX_TOKENS,
        "query_rewrite_enable_thinking": config.QUERY_REWRITE_ENABLE_THINKING,
        "vector_store": config.VECTOR_STORE_PATH,
        "default_top_k": config.DEFAULT_TOP_K,
        "use_hybrid_retrieval": config.USE_HYBRID_RETRIEVAL,
        "use_query_rewrite": config.USE_QUERY_REWRITE,
        "use_reranker": config.USE_RERANKER,
        "use_cot_prompt": config.USE_COT_PROMPT,
        "use_adaptive_retrieval": config.USE_ADAPTIVE_RETRIEVAL,
        "max_concurrent": config.CONCURRENCY.max_concurrent,
        "rpm_limit": config.CONCURRENCY.rpm_limit,
        "bm25_cache_path": str(bm25_cache_path),
    }

    reranker_stats = pipeline.reranker.get_stats() if pipeline.reranker else {}
    if config.USE_RERANKER and not reranker_stats.get("cross_encoder_available", False):
        print("[warn] Cross-Encoder failed to load; reranking is currently bypassed")

    print("[OK] Enhanced RAG Pipeline initialized")

    # We no longer evaluate the development set in this script.
    progress_mgr.clear_checkpoint(script_name="enhanced_eval_dev")

    # ============================================================
    # Evaluate on Test Set
    # ============================================================

    print(f"\n{'=' * 60}")
    print("Evaluating on Test Set")
    print(f"{'=' * 60}")

    # Check if we need to resume test set evaluation
    resume_info_test = get_compatible_resume_info(
        progress_mgr,
        "enhanced_eval_test",
        len(test_set),
    )

    if resume_info_test:
        print(
            f"\n🔄 Resuming test set evaluation from question {resume_info_test['start_from'] + 1}"
        )

    test_results = await evaluate_with_pipeline_async(
        pipeline,
        test_set,
        top_k=config.DEFAULT_TOP_K,
        dataset_name="Test Set",
        progress_mgr=progress_mgr,
        start_from=resume_info_test["start_from"] if resume_info_test else 0,
        initial_results=resume_info_test["results"] if resume_info_test else None,
        initial_correct=resume_info_test["correct_count"] if resume_info_test else 0,
        initial_total=resume_info_test["total_count"] if resume_info_test else 0,
        initial_elapsed=resume_info_test["elapsed_time"] if resume_info_test else 0.0,
        script_name="enhanced_eval_test",
        artifact_paths=artifact_paths,
        live_config=live_config,
    )

    # Clear checkpoint after successful completion
    progress_mgr.clear_checkpoint(script_name="enhanced_eval_test")

    # ============================================================
    # Save Results
    # ============================================================

    paths = progress_mgr.write_final_results(
        artifact_paths=artifact_paths,
        run_name="ENHANCED_RAG",
        evaluation_type="ENHANCED_RAG",
        config=live_config,
        stage_results={"test_set_evaluation": test_results},
    )

    # ============================================================
    # Print Final Summary
    # ============================================================

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n📊 Final Results:")
    print(f"  Test Set Accuracy: {test_results['accuracy']:.4f}")
    print(
        f"\n⏱️  Evaluation Time: {test_results['elapsed_time']:.1f}s "
        f"({test_results['questions_per_second']:.2f} questions/second)"
    )

    print(f"\n{'=' * 60}")
    print("Optimization Summary:")
    print(f"{'=' * 60}")
    print("✓ Phase 1: Hybrid Retrieval, Query Rewrite, Prompt Optimization")
    print("✓ Phase 2: Semantic Chunking, Metadata Enhancement, Reranking")
    print(f"Results JSON: {paths['json']}")
    print(f"Summary TXT: {paths['summary']}")
    print(f"{'=' * 60}")


def main():
    """CLI entrypoint."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
