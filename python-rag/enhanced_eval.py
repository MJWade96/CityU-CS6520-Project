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
import re
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
from app.rag.data_paths import (
    EVALUATION_DIR,
    EVALUATION_RESULTS_DIR,
    FAISS_INDEX_DIR,
)
from app.rag.embeddings import resolve_embedding_runtime
from app.rag.eval_shared import (
    ConcurrencyConfig,
    EvaluationLLMConfig,
    RateLimiter,
    build_medical_eval_prompt,
    create_async_client,
    extract_answer,
    get_correct_answer_letter,
    get_qwen_completion_kwargs,
    get_qwen_langchain_kwargs,
    load_questions as load_shared_questions,
    parse_optional_bool_env,
    split_questions,
)


# ============================================================
# Configuration
# ============================================================


def _env_flag(name: str, default: bool) -> bool:
    """Resolve a boolean env var with a concrete fallback."""
    value = parse_optional_bool_env(name, default=default)
    return default if value is None else value


class EnhancedEvaluationConfig:
    """Enhanced evaluation configuration"""

    # Dataset split
    # Match evaluate_no_rag.py and complete_eval.py:
    # dev uses questions[:300], test uses questions[300:].
    DEV_SET_SIZE = 300
    TEST_SET_SIZE = None

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
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE = os.getenv("RAG_EMBEDDING_DEVICE", "auto")

    # Retrieval configuration
    TOP_K_VALUES = [1, 3, 5, 10]
    DEFAULT_TOP_K = 5

    # Optimization flags
    USE_HYBRID_RETRIEVAL = True
    USE_QUERY_REWRITE = True
    USE_LLM_QUERY_REWRITE = _env_flag("RAG_ENHANCED_USE_LLM_QUERY_REWRITE", True)
    LLM_QUERY_REWRITE_MODE = os.getenv(
        "RAG_ENHANCED_LLM_QUERY_REWRITE_MODE",
        "auto",
    ).strip().lower()
    LLM_QUERY_REWRITE_AUTO_MAX_CHARS = max(
        1,
        int(os.getenv("RAG_ENHANCED_LLM_QUERY_REWRITE_AUTO_MAX_CHARS", "160")),
    )
    LLM_QUERY_REWRITE_AUTO_MAX_WORDS = max(
        1,
        int(os.getenv("RAG_ENHANCED_LLM_QUERY_REWRITE_AUTO_MAX_WORDS", "24")),
    )
    USE_RERANKER = True
    RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-large")
    RERANKER_DEVICE = os.getenv("RAG_RERANKER_DEVICE", "auto")
    USE_COT_PROMPT = False
    USE_ADAPTIVE_RETRIEVAL = False
    CONCURRENCY = ConcurrencyConfig(
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
    PROGRESS_SAVE_EVERY = max(1, int(os.getenv("RAG_ENHANCED_EVAL_SAVE_EVERY", "5")))
    PROGRESS_PRINT_EVERY = max(
        1,
        int(os.getenv("RAG_ENHANCED_EVAL_PRINT_EVERY", "5")),
    )
    HEARTBEAT_ENABLED = _env_flag("RAG_ENHANCED_EVAL_HEARTBEAT_ENABLED", True)
    HEARTBEAT_INTERVAL_SECONDS = max(
        1.0,
        float(os.getenv("RAG_ENHANCED_EVAL_HEARTBEAT_INTERVAL_SECONDS", "15")),
    )
    QUESTION_START_LOG_ENABLED = _env_flag(
        "RAG_ENHANCED_EVAL_QUESTION_START_LOG_ENABLED",
        True,
    )
    QUESTION_START_LOG_PREVIEW_CHARS = max(
        20,
        int(os.getenv("RAG_ENHANCED_EVAL_QUESTION_START_LOG_PREVIEW_CHARS", "120")),
    )
    IN_FLIGHT_MULTIPLIER = max(
        1,
        int(os.getenv("RAG_ENHANCED_EVAL_IN_FLIGHT_MULTIPLIER", "2")),
    )

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
        self.llm_config = llm_config
        self.completion_kwargs = get_qwen_completion_kwargs(llm_config)
        self.async_client = create_async_client(llm_config)
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **get_qwen_langchain_kwargs(llm_config),
        )

    def _build_prompt(
        self,
        question: str,
        contexts: List[str],
        options: Optional[List[str]] = None,
    ) -> str:
        """Build the shared evaluation prompt for sync/async generation."""
        return build_medical_eval_prompt(
            question=question,
            options=options or [],
            context="\n\n".join(
                f"[{index + 1}] {context}"
                for index, context in enumerate(contexts)
            ),
        )

    def generate(
        self,
        question: str,
        contexts: List[str],
        options: Optional[List[str]] = None,
    ) -> str:
        """Generate an answer using the shared evaluation prompt."""
        prompt = self._build_prompt(question, contexts, options)

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    async def generate_async(
        self,
        question: str,
        contexts: List[str],
        options: Optional[List[str]] = None,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        api_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> str:
        """Generate an answer asynchronously for higher evaluation throughput."""
        prompt = self._build_prompt(question, contexts, options)

        try:
            if api_semaphore:
                async with api_semaphore:
                    if rate_limiter:
                        await rate_limiter.acquire()
                    completion = await self.async_client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        **self.completion_kwargs,
                    )
            else:
                if rate_limiter:
                    await rate_limiter.acquire()
                completion = await self.async_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    **self.completion_kwargs,
                )

            return (
                completion.choices[0].message.content
                or completion.choices[0].message.reasoning_content
                or ""
            )
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
            use_dict=config.USE_QUERY_REWRITE,
            use_llm=config.USE_QUERY_REWRITE and config.USE_LLM_QUERY_REWRITE,
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
            cross_encoder_model=config.RERANKER_MODEL,
            cross_encoder_device=config.RERANKER_DEVICE,
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
        dict_rewriter = getattr(self.query_rewriter, "dict_rewriter", None)
        self._rewrite_abbreviation_patterns = tuple(
            re.compile(rf"\b{re.escape(abbr)}\b", flags=re.IGNORECASE)
            for abbr in getattr(dict_rewriter, "abbreviations", {})
        )
        self._rewrite_chinese_terms = tuple(
            getattr(dict_rewriter, "chinese_terms", {}).keys()
        )

    def _should_use_llm_query_rewrite(self, query: str) -> bool:
        """Use LLM query rewrite selectively to cut extra API latency."""
        if not (self.config.USE_QUERY_REWRITE and self.config.USE_LLM_QUERY_REWRITE):
            return False

        mode = self.config.LLM_QUERY_REWRITE_MODE
        if mode == "always":
            return True
        if mode == "never":
            return False

        normalized = (query or "").strip()
        if not normalized:
            return False

        if any("\u4e00" <= ch <= "\u9fff" for ch in normalized):
            return True

        lowered = normalized.lower()
        if any(pattern.search(lowered) for pattern in self._rewrite_abbreviation_patterns):
            return True

        if any(term in normalized for term in self._rewrite_chinese_terms):
            return True

        if len(normalized) > self.config.LLM_QUERY_REWRITE_AUTO_MAX_CHARS:
            return False

        if len(normalized.split()) > self.config.LLM_QUERY_REWRITE_AUTO_MAX_WORDS:
            return False

        return True

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
            primary_query, all_queries = self.query_rewriter.rewrite_with_options(
                query,
                mode="single",
                use_llm=self._should_use_llm_query_rewrite(query),
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

    async def retrieve_async(
        self,
        query: str,
        top_k: int = 5,
        use_rewrite: bool = True,
        use_rerank: bool = True,
        use_adaptive: bool = False,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        api_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> List[Tuple[Document, float]]:
        """Async retrieval path that keeps remote calls on the async client."""
        if use_rewrite:
            primary_query, _ = await self.query_rewriter.arewrite(
                query,
                mode="single",
                rate_limiter=rate_limiter,
                api_semaphore=api_semaphore,
                use_llm=self._should_use_llm_query_rewrite(query),
            )
        else:
            primary_query = query

        if use_adaptive:
            results = await asyncio.to_thread(
                self.adaptive_retriever.search,
                primary_query,
                top_k * 2,
            )
        else:
            results = await asyncio.to_thread(
                self.hybrid_retriever.search,
                primary_query,
                top_k * 2,
                self.config.USE_HYBRID_RETRIEVAL,
            )

        if use_rerank and self.reranker:
            results = await asyncio.to_thread(self.reranker.rerank, primary_query, results)

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

    async def answer_async(
        self,
        query: str,
        options: Optional[List[str]] = None,
        top_k: int = 5,
        use_rewrite: bool = True,
        use_rerank: bool = True,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        api_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Dict[str, Any]:
        """Async end-to-end RAG flow for higher evaluation throughput."""
        results = await self.retrieve_async(
            query,
            top_k=top_k,
            use_rewrite=use_rewrite,
            use_rerank=use_rerank,
            rate_limiter=rate_limiter,
            api_semaphore=api_semaphore,
        )
        contexts = [doc.page_content for doc, score in results]
        response = await self.llm_generator.generate_async(
            query,
            contexts,
            options,
            rate_limiter=rate_limiter,
            api_semaphore=api_semaphore,
        )
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

    questions = load_shared_questions(question_file)

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
    embedding_runtime = resolve_embedding_runtime(
        config.VECTOR_STORE_PATH,
        default_model=config.EMBEDDING_MODEL,
        preferred_device=config.EMBEDDING_DEVICE,
    )
    recorded_model = embedding_runtime["recorded_model"]
    if recorded_model:
        print(f"  Index embedding model: {recorded_model}")
    if recorded_model and recorded_model != embedding_runtime["model_name"]:
        print(
            f"[warn] Runtime embedding model '{embedding_runtime['model_name']}' "
            f"differs from index metadata '{recorded_model}'"
        )
    print(f"  Runtime embedding model: {embedding_runtime['model_name']}")
    print(f"  Runtime embedding device: {embedding_runtime['device']}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_runtime["model_name"],
        model_kwargs={"device": embedding_runtime["device"]},
        encode_kwargs={"normalize_embeddings": True},
    )
    config.RESOLVED_EMBEDDING_MODEL = embedding_runtime["model_name"]
    config.RESOLVED_EMBEDDING_DEVICE = embedding_runtime["device"]
    config.INDEX_EMBEDDING_MODEL = recorded_model

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
        "embedding_model": getattr(
            pipeline.config,
            "RESOLVED_EMBEDDING_MODEL",
            pipeline.config.EMBEDDING_MODEL,
        ),
        "embedding_device": getattr(
            pipeline.config,
            "RESOLVED_EMBEDDING_DEVICE",
            pipeline.config.EMBEDDING_DEVICE,
        ),
        "query_rewrite_provider": pipeline.config.QUERY_REWRITE_PROVIDER,
        "query_rewrite_model": pipeline.config.QUERY_REWRITE_MODEL,
        "query_rewrite_temperature": pipeline.config.QUERY_REWRITE_TEMPERATURE,
        "query_rewrite_max_tokens": pipeline.config.QUERY_REWRITE_MAX_TOKENS,
        "query_rewrite_enable_thinking": pipeline.config.QUERY_REWRITE_ENABLE_THINKING,
        "llm_query_rewrite": pipeline.config.USE_LLM_QUERY_REWRITE,
        "hybrid_retrieval": pipeline.config.USE_HYBRID_RETRIEVAL,
        "query_rewrite": pipeline.config.USE_QUERY_REWRITE,
        "reranker": pipeline.config.USE_RERANKER,
        "reranker_model": pipeline.config.RERANKER_MODEL,
        "reranker_device": reranker_stats.get("cross_encoder_device"),
        "reranker_available": reranker_stats.get("cross_encoder_available", False),
        "max_concurrent": pipeline.config.CONCURRENCY.max_concurrent,
        "rpm_limit": pipeline.config.CONCURRENCY.rpm_limit,
        "progress_save_every": pipeline.config.PROGRESS_SAVE_EVERY,
        "progress_print_every": pipeline.config.PROGRESS_PRINT_EVERY,
        "heartbeat_enabled": pipeline.config.HEARTBEAT_ENABLED,
        "heartbeat_interval_seconds": pipeline.config.HEARTBEAT_INTERVAL_SECONDS,
        "question_start_log_enabled": pipeline.config.QUESTION_START_LOG_ENABLED,
        "question_start_log_preview_chars": (
            pipeline.config.QUESTION_START_LOG_PREVIEW_CHARS
        ),
        "in_flight_multiplier": pipeline.config.IN_FLIGHT_MULTIPLIER,
        "llm_query_rewrite_mode": pipeline.config.LLM_QUERY_REWRITE_MODE,
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
    """Evaluate using the enhanced pipeline with rolling concurrency."""
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
    max_in_flight = max(1, batch_size * pipeline.config.IN_FLIGHT_MULTIPLIER)
    persist_every = max(1, pipeline.config.PROGRESS_SAVE_EVERY)
    print_every = max(1, pipeline.config.PROGRESS_PRINT_EVERY)
    heartbeat_enabled = pipeline.config.HEARTBEAT_ENABLED
    heartbeat_interval = max(1.0, pipeline.config.HEARTBEAT_INTERVAL_SECONDS)
    question_start_log_enabled = pipeline.config.QUESTION_START_LOG_ENABLED
    question_start_log_preview_chars = max(
        20,
        pipeline.config.QUESTION_START_LOG_PREVIEW_CHARS,
    )
    api_semaphore = asyncio.Semaphore(batch_size)
    rate_limiter = RateLimiter(
        requests_per_second=pipeline.config.CONCURRENCY.requests_per_second,
        burst=batch_size,
    )
    last_heartbeat_at = time.time()

    if heartbeat_enabled:
        print(
            f"Heartbeat enabled: every {heartbeat_interval:.0f}s "
            f"(shows activity even when ordered progress is waiting)"
        )

    def emit_heartbeat(reason: str) -> None:
        """Print a lightweight status line when long-running work looks stalled."""
        nonlocal last_heartbeat_at

        if not heartbeat_enabled:
            return

        now = time.time()
        committed = total
        completed_any = total + len(buffered_results)
        buffered = len(buffered_results)
        waiting_on = next_commit_index + 1 if buffered else None
        waiting_suffix = f", waiting_on_q={waiting_on}" if waiting_on else ""
        print(
            "  heartbeat: "
            f"committed={committed}/{len(questions)}, "
            f"completed={completed_any}/{len(questions)}, "
            f"in_flight={len(in_flight)}, "
            f"buffered={buffered}, "
            f"elapsed={now - start_time:.1f}s, "
            f"reason={reason}{waiting_suffix}"
        )
        last_heartbeat_at = now

    def format_question_preview(text: str) -> str:
        """Collapse whitespace so start logs stay readable on long prompts."""
        preview = re.sub(r"\s+", " ", (text or "")).strip()
        if len(preview) <= question_start_log_preview_chars:
            return preview
        return preview[: question_start_log_preview_chars - 3].rstrip() + "..."

    async def evaluate_item(
        question_index: int,
        q: Dict[str, Any],
    ) -> Dict[str, Any]:
        question_text = q.get("question", "")
        options = q.get("options", [])
        correct_answer_letter = get_correct_answer_letter(q)
        uses_llm_rewrite = (
            pipeline.config.USE_QUERY_REWRITE
            and pipeline.config.USE_LLM_QUERY_REWRITE
            and pipeline._should_use_llm_query_rewrite(question_text)
        )

        if question_start_log_enabled:
            print(
                "  start: "
                f"q={question_index + 1}/{len(questions)}, "
                f"chars={len(question_text)}, "
                f"options={len(options)}, "
                f"hybrid={pipeline.config.USE_HYBRID_RETRIEVAL}, "
                f"rewrite={pipeline.config.USE_QUERY_REWRITE}, "
                f"llm_rewrite={uses_llm_rewrite}, "
                f"reranker={pipeline.config.USE_RERANKER}, "
                f"top_k={top_k}, "
                f"preview=\"{format_question_preview(question_text)}\""
            )

        try:
            result = await pipeline.answer_async(
                question_text,
                options,
                top_k,
                pipeline.config.USE_QUERY_REWRITE,
                pipeline.config.USE_RERANKER,
                rate_limiter=rate_limiter,
                api_semaphore=api_semaphore,
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

    indexed_questions = iter(enumerate(questions_to_process, start=start_from))
    in_flight: Dict[asyncio.Task, Tuple[int, Dict[str, Any]]] = {}
    buffered_results: Dict[int, Dict[str, Any]] = {}
    next_commit_index = start_from

    async def run_item(
        question_index: int,
        item: Dict[str, Any],
    ) -> Tuple[int, Dict[str, Any]]:
        return question_index, await evaluate_item(question_index, item)

    def schedule_next() -> bool:
        try:
            question_index, item = next(indexed_questions)
        except StopIteration:
            return False

        task = asyncio.create_task(run_item(question_index, item))
        in_flight[task] = (question_index, item)
        return True

    for _ in range(min(max_in_flight, len(questions_to_process))):
        if not schedule_next():
            break

    while in_flight:
        done, _ = await asyncio.wait(
            tuple(in_flight.keys()),
            timeout=heartbeat_interval if heartbeat_enabled else None,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            emit_heartbeat("awaiting_completion")
            continue

        for task in done:
            question_index, item = in_flight.pop(task)
            try:
                completed_index, evaluation_result = await task
            except Exception as e:
                completed_index = question_index
                correct_answer_letter = get_correct_answer_letter(item)
                evaluation_result = {
                    "question": item.get("question", ""),
                    "options": item.get("options", []),
                    "correct_answer": correct_answer_letter,
                    "predicted_answer": None,
                    "is_correct": False,
                    "response": f"Error generating answer: {str(e)}",
                    "retrieved_docs": 0,
                    "error": str(e),
                }
            buffered_results[completed_index] = evaluation_result
            schedule_next()

        committed_before = total

        while next_commit_index in buffered_results:
            evaluation_result = buffered_results.pop(next_commit_index)
            processed_questions = next_commit_index + 1

            if evaluation_result["is_correct"]:
                correct += 1
            total += 1
            results.append(evaluation_result)
            next_commit_index += 1

            elapsed = time.time() - start_time

            if evaluation_result.get("error"):
                print(f"  ERROR on question {processed_questions}: {evaluation_result['error']}")

            if progress_mgr:
                should_print = (
                    processed_questions == len(questions)
                    or evaluation_result.get("error") is not None
                    or processed_questions % print_every == 0
                )
                if should_print:
                    progress_mgr.print_progress(
                        run_name="ENHANCED_RAG",
                        dataset_name=dataset_name,
                        processed_questions=processed_questions,
                        total_questions=len(questions),
                        correct_count=correct,
                        elapsed_time=elapsed,
                    )

                should_persist = (
                    processed_questions == len(questions)
                    or evaluation_result.get("error") is not None
                    or processed_questions % persist_every == 0
                )
                if should_persist:
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

        if (
            heartbeat_enabled
            and buffered_results
            and total == committed_before
            and time.time() - last_heartbeat_at >= heartbeat_interval
        ):
            emit_heartbeat("waiting_for_ordered_commit")

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

    dev_set, test_set = split_questions(
        questions,
        config.DEV_SET_SIZE,
        config.TEST_SET_SIZE,
    )
    test_start_index = len(dev_set)

    print(f"\nEvaluation Scope:")
    print(f"  Only evaluating test set")
    print(f"  Dev set size (aligned, not evaluated here): {len(dev_set)} questions")
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
    print(
        f"  Embedding: "
        f"{getattr(config, 'RESOLVED_EMBEDDING_MODEL', config.EMBEDDING_MODEL)} "
        f"on {getattr(config, 'RESOLVED_EMBEDDING_DEVICE', config.EMBEDDING_DEVICE)}"
    )
    print(f"  Hybrid Retrieval: {config.USE_HYBRID_RETRIEVAL}")
    print(f"  Query Rewrite: {config.USE_QUERY_REWRITE}")
    print(f"  LLM Query Rewrite: {config.USE_LLM_QUERY_REWRITE}")
    print(f"  LLM Query Rewrite Mode: {config.LLM_QUERY_REWRITE_MODE}")
    print(f"  Reranker: {config.USE_RERANKER}")
    print(f"  Reranker Model: {config.RERANKER_MODEL}")
    print(f"  Reranker Device: {config.RERANKER_DEVICE}")
    print(f"  CoT Prompting: {config.USE_COT_PROMPT}")
    print(f"  Adaptive Retrieval: {config.USE_ADAPTIVE_RETRIEVAL}")
    print(f"  Max Concurrent: {config.CONCURRENCY.max_concurrent}")
    print(f"  In-Flight Multiplier: {config.IN_FLIGHT_MULTIPLIER}")
    print(f"  Progress Save Every: {config.PROGRESS_SAVE_EVERY} questions")
    print(f"  Progress Print Every: {config.PROGRESS_PRINT_EVERY} questions")
    print(f"  Heartbeat Enabled: {config.HEARTBEAT_ENABLED}")
    print(f"  Heartbeat Interval: {config.HEARTBEAT_INTERVAL_SECONDS:.0f}s")
    print(f"  Question Start Logs: {config.QUESTION_START_LOG_ENABLED}")
    print(
        f"  Question Start Preview Chars: "
        f"{config.QUESTION_START_LOG_PREVIEW_CHARS}"
    )
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
        "dev_set_size": len(dev_set),
        "test_set_size": len(test_set),
        "test_question_start_index": test_start_index,
        "test_question_end_index": test_start_index + len(test_set) - 1,
        "embedding_model": getattr(
            config,
            "RESOLVED_EMBEDDING_MODEL",
            config.EMBEDDING_MODEL,
        ),
        "embedding_device": getattr(
            config,
            "RESOLVED_EMBEDDING_DEVICE",
            config.EMBEDDING_DEVICE,
        ),
        "index_embedding_model": getattr(config, "INDEX_EMBEDDING_MODEL", None),
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": config.LLM_MODEL,
        "query_rewrite_provider": config.QUERY_REWRITE_PROVIDER,
        "query_rewrite_model": config.QUERY_REWRITE_MODEL,
        "query_rewrite_temperature": config.QUERY_REWRITE_TEMPERATURE,
        "query_rewrite_max_tokens": config.QUERY_REWRITE_MAX_TOKENS,
        "query_rewrite_enable_thinking": config.QUERY_REWRITE_ENABLE_THINKING,
        "llm_query_rewrite": config.USE_LLM_QUERY_REWRITE,
        "llm_query_rewrite_mode": config.LLM_QUERY_REWRITE_MODE,
        "vector_store": config.VECTOR_STORE_PATH,
        "default_top_k": config.DEFAULT_TOP_K,
        "use_hybrid_retrieval": config.USE_HYBRID_RETRIEVAL,
        "use_query_rewrite": config.USE_QUERY_REWRITE,
        "use_reranker": config.USE_RERANKER,
        "reranker_model": config.RERANKER_MODEL,
        "reranker_device": config.RERANKER_DEVICE,
        "use_cot_prompt": config.USE_COT_PROMPT,
        "use_adaptive_retrieval": config.USE_ADAPTIVE_RETRIEVAL,
        "max_concurrent": config.CONCURRENCY.max_concurrent,
        "in_flight_multiplier": config.IN_FLIGHT_MULTIPLIER,
        "rpm_limit": config.CONCURRENCY.rpm_limit,
        "progress_save_every": config.PROGRESS_SAVE_EVERY,
        "progress_print_every": config.PROGRESS_PRINT_EVERY,
        "heartbeat_enabled": config.HEARTBEAT_ENABLED,
        "heartbeat_interval_seconds": config.HEARTBEAT_INTERVAL_SECONDS,
        "question_start_log_enabled": config.QUESTION_START_LOG_ENABLED,
        "question_start_log_preview_chars": (
            config.QUESTION_START_LOG_PREVIEW_CHARS
        ),
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
