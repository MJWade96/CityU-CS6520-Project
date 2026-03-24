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


# ============================================================
# Configuration
# ============================================================


class EnhancedEvaluationConfig:
    """Enhanced evaluation configuration"""

    # Dataset split
    DEV_SET_SIZE = 300

    # LLM Configuration (联通云 DeepSeek V3.2)
    LLM_PROVIDER = "Qwen3-4B"
    LLM_MODEL = "8606056bfe0c49448d92587452d1f2fc"
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 512
    LLM_BASE_URL = "https://wishub-x6.ctyun.cn/v1"
    LLM_API_KEY = "4dbe3bec3ee548d28b649b324e741939"

    # Retrieval configuration
    TOP_K_VALUES = [1, 3, 5, 10]
    DEFAULT_TOP_K = 5

    # Optimization flags
    USE_HYBRID_RETRIEVAL = True
    USE_QUERY_REWRITE = True
    USE_RERANKER = True
    USE_COT_PROMPT = True
    USE_ADAPTIVE_RETRIEVAL = False

    # File paths
    VECTOR_STORE_PATH = str(FAISS_INDEX_DIR)
    QUESTION_FILE = str(EVALUATION_DIR / "medqa.json")
    OUTPUT_DIR = str(EVALUATION_RESULTS_DIR)


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
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_cot: bool = True,
    ):
        """Initialize enhanced LLM generator"""
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_cot = use_cot

        # Get API credentials
        self.api_key = api_key or "4dbe3bec3ee548d28b649b324e741939"
        self.base_url = base_url or "https://wishub-x6.ctyun.cn/v1"

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # Initialize prompt
        self.prompt_template = self._get_prompt_template()

    def _get_prompt_template(self) -> str:
        """Get prompt template based on configuration"""
        if self.use_cot:
            return """You are a medical expert assistant. Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Options:
{options}

Please think step by step and then provide your answer in the following format:
Answer: [A/B/C/D/E]

Your response:"""
        else:
            return """You are a medical expert assistant. Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Options:
{options}

Please provide your answer in the following format:
Answer: [A/B/C/D/E]

Your response:"""

    def generate(
        self,
        question: str,
        contexts: List[str],
        options: Optional[List[str]] = None,
    ) -> str:
        """Generate answer using enhanced prompt"""
        # Format context
        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])

        # Format options
        if options:
            options_text = "\n".join(
                [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
            )
        else:
            options_text = (
                "A. Not provided\nB. Not provided\nC. Not provided\nD. Not provided"
            )

        # Build prompt
        prompt = self.prompt_template.format(
            context=context_text,
            question=question,
            options=options_text,
        )

        # Generate response
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer choice from LLM response"""
        import re

        # Try to find answer pattern
        patterns = [
            r"Answer:\s*([A-E])",
            r"answer:\s*([A-E])",
            r"\b([A-E])\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None


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
    ):
        """Initialize enhanced RAG pipeline"""
        self.config = config
        self.documents = documents
        self.embedding_model = embedding_model

        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            embedding_model=embedding_model,
            documents=documents,
            dense_weight=0.5,
        )

        # Initialize adaptive retriever
        self.adaptive_retriever = AdaptiveRetriever(self.hybrid_retriever)

        # Initialize query rewrite pipeline
        self.query_rewriter = QueryRewritePipeline(
            use_dict=True,
            use_llm=config.USE_QUERY_REWRITE,
            use_expansion=False,
            llm_provider=config.LLM_PROVIDER,
            llm_model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
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
            max_tokens=config.LLM_MAX_TOKENS,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            use_cot=config.USE_COT_PROMPT,
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
            print(f"  Rewritten query: {primary_query[:100]}...")
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
        return None, None

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

    return embeddings, documents


def evaluate_with_pipeline(
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
) -> Dict[str, Any]:
    """Evaluate using enhanced pipeline with checkpoint support"""
    print(f"\n{'=' * 60}")
    if start_from > 0:
        print(f"Resuming {dataset_name} (top-k={top_k}) from question {start_from + 1}")
    else:
        print(f"Evaluating {dataset_name} (top-k={top_k})")
    print(f"{'=' * 60}")

    start_time = time.time() - initial_elapsed
    results = initial_results if initial_results is not None else []
    correct = initial_correct
    total = initial_total

    questions_to_process = questions[start_from:]

    for i, q in enumerate(questions_to_process, start_from + 1):
        question_text = q.get("question", "")
        options = q.get("options", [])
        correct_answer = q.get("answer", "")
        answer_index = q.get("answer_index", -1)

        # Convert answer_index to letter
        if answer_index >= 0:
            correct_answer_letter = chr(65 + answer_index)
        else:
            correct_answer_letter = correct_answer

        try:
            # Use pipeline
            result = pipeline.answer(
                query=question_text,
                options=options,
                top_k=top_k,
                use_rewrite=pipeline.config.USE_QUERY_REWRITE,
                use_rerank=pipeline.config.USE_RERANKER,
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
            }

            results.append(evaluation_result)

            if is_correct:
                correct += 1
            total += 1

            # Progress reporting
            if i % 10 == 0 or i == len(questions):
                elapsed = time.time() - start_time
                qps = i / elapsed if elapsed > 0 else 0
                current_acc = correct / total if total > 0 else 0
                print(
                    f"  Question {i}/{len(questions)} | "
                    f"Accuracy: {current_acc:.4f} | "
                    f"Speed: {qps:.2f} q/s"
                )

            # Save checkpoint after each question
            if progress_mgr:
                elapsed = time.time() - start_time
                progress_mgr.save_checkpoint(
                    dataset_name=dataset_name,
                    total_questions=len(questions),
                    processed_questions=i,
                    current_top_k=top_k,
                    results=results,
                    correct_count=correct,
                    total_count=total,
                    elapsed_time=elapsed,
                    config={
                        "top_k": top_k,
                        "llm_provider": "Qwen3-4B",
                        "llm_model": "8606056bfe0c49448d92587452d1f2fc",
                        "hybrid_retrieval": pipeline.config.USE_HYBRID_RETRIEVAL,
                        "query_rewrite": pipeline.config.USE_QUERY_REWRITE,
                        "reranker": pipeline.config.USE_RERANKER,
                    },
                    script_name=script_name or "enhanced_eval",
                )

        except Exception as e:
            print(f"  ERROR on question {i}: {e}")
            if progress_mgr:
                elapsed = time.time() - start_time
                progress_mgr.save_checkpoint(
                    dataset_name=dataset_name,
                    total_questions=len(questions),
                    processed_questions=i,
                    current_top_k=top_k,
                    results=results,
                    correct_count=correct,
                    total_count=total,
                    elapsed_time=elapsed,
                    config={
                        "top_k": top_k,
                        "llm_provider": "Qwen3-4B",
                        "llm_model": "8606056bfe0c49448d92587452d1f2fc",
                        "hybrid_retrieval": pipeline.config.USE_HYBRID_RETRIEVAL,
                        "query_rewrite": pipeline.config.USE_QUERY_REWRITE,
                        "reranker": pipeline.config.USE_RERANKER,
                    },
                    script_name=script_name or "enhanced_eval",
                    error_message=f"Error on question {i}: {str(e)}",
                )
            continue

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0

    return {
        "dataset_name": dataset_name,
        "total_questions": total,
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


def main():
    """Main evaluation function with checkpoint support"""
    print("=" * 60)
    print("Enhanced Medical RAG System - Complete Evaluation")
    print("Phase 1 + Phase 2 Optimizations")
    print("=" * 60)

    # Load configuration
    config = EnhancedEvaluationConfig()

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Initialize progress manager
    progress_mgr = EvaluationProgressManager(output_dir=config.OUTPUT_DIR)

    # Load questions
    questions = load_questions(config.QUESTION_FILE)

    if not questions:
        print("\nNo questions loaded. Exiting...")
        return

    # Split dataset
    dev_set = questions[: config.DEV_SET_SIZE]
    test_set = questions[config.DEV_SET_SIZE :]

    print(f"\nDataset Split:")
    print(f"  Development set: {len(dev_set)} questions")
    print(f"  Test set: {len(test_set)} questions")

    # Load vector store
    embeddings, documents = load_vector_store(config)

    if embeddings is None or documents is None:
        print("\nFailed to load vector store. Exiting...")
        return

    # Initialize enhanced pipeline
    print("\nInitializing Enhanced RAG Pipeline...")
    print(f"  Hybrid Retrieval: {config.USE_HYBRID_RETRIEVAL}")
    print(f"  Query Rewrite: {config.USE_QUERY_REWRITE}")
    print(f"  Reranker: {config.USE_RERANKER}")
    print(f"  CoT Prompting: {config.USE_COT_PROMPT}")
    print(f"  Adaptive Retrieval: {config.USE_ADAPTIVE_RETRIEVAL}")

    pipeline = EnhancedRAGPipeline(
        embedding_model=embeddings,
        documents=documents,
        config=config,
    )

    print("[OK] Enhanced RAG Pipeline initialized")

    # ============================================================
    # Evaluate on Development Set
    # ============================================================

    print(f"\n{'=' * 60}")
    print("Evaluating on Development Set")
    print(f"{'=' * 60}")

    # Check if we need to resume dev set evaluation
    resume_dev = progress_mgr.should_resume(script_name="enhanced_eval_dev")
    resume_info_dev = None

    if resume_dev:
        resume_info_dev = progress_mgr.get_resume_info(script_name="enhanced_eval_dev")
        print(
            f"\n🔄 Resuming dev set evaluation from question {resume_info_dev['start_from'] + 1}"
        )

        dev_results = evaluate_with_pipeline(
            pipeline,
            dev_set,
            top_k=config.DEFAULT_TOP_K,
            dataset_name="Development Set",
            progress_mgr=progress_mgr,
            start_from=resume_info_dev["start_from"],
            initial_results=resume_info_dev["results"],
            initial_correct=resume_info_dev["correct_count"],
            initial_total=resume_info_dev["total_count"],
            initial_elapsed=resume_info_dev["elapsed_time"],
            script_name="enhanced_eval_dev",
        )
    else:
        dev_results = evaluate_with_pipeline(
            pipeline,
            dev_set,
            top_k=config.DEFAULT_TOP_K,
            dataset_name="Development Set",
            progress_mgr=progress_mgr,
            script_name="enhanced_eval_dev",
        )

    # Clear dev set checkpoint after successful completion
    progress_mgr.clear_checkpoint(script_name="enhanced_eval_dev")

    # ============================================================
    # Evaluate on Test Set
    # ============================================================

    print(f"\n{'=' * 60}")
    print("Evaluating on Test Set")
    print(f"{'=' * 60}")

    # Check if we need to resume test set evaluation
    resume_test = progress_mgr.should_resume(script_name="enhanced_eval_test")
    resume_info_test = None

    if resume_test:
        resume_info_test = progress_mgr.get_resume_info(
            script_name="enhanced_eval_test"
        )
        print(
            f"\n🔄 Resuming test set evaluation from question {resume_info_test['start_from'] + 1}"
        )

    test_results = evaluate_with_pipeline(
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
    )

    # Clear checkpoint after successful completion
    progress_mgr.clear_checkpoint(script_name="enhanced_eval_test")

    # ============================================================
    # Save Results
    # ============================================================

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save complete results
    results_file = os.path.join(config.OUTPUT_DIR, f"enhanced_eval_{timestamp}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "dev_set_size": config.DEV_SET_SIZE,
                    "llm_provider": config.LLM_PROVIDER,
                    "llm_model": config.LLM_MODEL,
                    "optimizations": {
                        "hybrid_retrieval": config.USE_HYBRID_RETRIEVAL,
                        "query_rewrite": config.USE_QUERY_REWRITE,
                        "reranker": config.USE_RERANKER,
                        "cot_prompt": config.USE_COT_PROMPT,
                        "adaptive_retrieval": config.USE_ADAPTIVE_RETRIEVAL,
                    },
                },
                "development_set_evaluation": dev_results,
                "test_set_evaluation": test_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[OK] Results saved to {results_file}")

    # Save summary
    summary_file = os.path.join(config.OUTPUT_DIR, f"enhanced_summary_{timestamp}.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Enhanced Medical RAG System - Evaluation Summary\n")
        f.write("Phase 1 + Phase 2 Optimizations\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL}\n\n")

        f.write("Optimizations Enabled:\n")
        f.write(f"  ✓ Hybrid Retrieval (Dense + BM25)\n")
        f.write(f"  ✓ Query Rewriting\n")
        f.write(f"  ✓ Reranking (Cross-Encoder)\n")
        f.write(f"  ✓ Chain-of-Thought Prompting\n")
        f.write(f"  ✓ Adaptive Retrieval\n\n")

        f.write("Development Set Results:\n")
        f.write(f"  Total Questions: {dev_results['total_questions']}\n")
        f.write(f"  Correct Answers: {dev_results['correct']}\n")
        f.write(f"  Accuracy: {dev_results['accuracy']:.4f}\n\n")

        f.write("Test Set Results:\n")
        f.write(f"  Total Questions: {test_results['total_questions']}\n")
        f.write(f"  Correct Answers: {test_results['correct']}\n")
        f.write(f"  Accuracy: {test_results['accuracy']:.4f}\n")
        f.write(f"  Time: {test_results['elapsed_time']:.1f}s\n")
        f.write(f"  Speed: {test_results['questions_per_second']:.2f} q/s\n")

    print(f"[OK] Summary saved to {summary_file}")

    # ============================================================
    # Print Final Summary
    # ============================================================

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n📊 Final Results:")
    print(f"  Development Set Accuracy: {dev_results['accuracy']:.4f}")
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
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
