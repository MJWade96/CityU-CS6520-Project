"""
Complete Medical RAG Evaluation System

严格按照规划文档实现：
1. 开发集 (Dev Set): 300 题 - 用于调试超参数（如 top-k）
2. 测试集 (Test Set): 剩余题目 - 用于最终评估
3. 核心指标：Accuracy（准确率）
4. 检索质量指标：Recall@k

完整 RAG 评估流程：
1. 检索阶段：使用问题从向量库中检索 top-k 个相关文档
2. 生成阶段：将问题 + 检索到的文档上下文发送给 LLM
3. 答案比较：LLM 生成的答案 vs 正确答案
4. 准确率计算：判断 LLM 生成的答案是否正确

Usage:
    python complete_eval.py
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
from langchain_core.prompts import PromptTemplate

from app.rag.vector_store import MedicalVectorStore


# ============================================================
# Configuration
# ============================================================


class EvaluationConfig:
    """评估配置"""

    # 获取脚本所在目录
    SCRIPT_DIR = Path(__file__).parent

    # 数据集划分
    DEV_SET_SIZE = 300  # 开发集题目数量

    # LLM 配置 (联通云 DeepSeek V3.2)
    LLM_PROVIDER = "deepseek"
    LLM_MODEL = "2656053fa69c4c2d89c5a691d9d737c3"  # DeepSeek V3.2
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 512
    LLM_BASE_URL = "https://wishub-x6.ctyun.cn/v1"  # 联通云 API 端点
    LLM_API_KEY = "6fcecb364d0647d2883e7f1d3f19d5b9"  # 联通云 API Key

    # 检索配置
    TOP_K_VALUES = [1, 3, 5, 10]  # 要测试的 top-k 值
    DEFAULT_TOP_K = 5

    # 文件路径 (使用绝对路径)
    # 使用临时目录避免中文路径问题
    VECTOR_STORE_PATH = r"C:\Users\MJWade\AppData\Local\Temp\medical_rag_faiss"
    QUESTION_FILE = str(SCRIPT_DIR / "data" / "evaluation" / "medqa.json")
    OUTPUT_DIR = str(SCRIPT_DIR / "results" / "evaluation")


# ============================================================
# Prompt Templates
# ============================================================

MEDICAL_RAG_PROMPT = PromptTemplate.from_template(
    """You are a medical expert assistant. Answer the following question based on the provided context. If the context does not contain enough information to answer the question, state that you cannot answer based on the given information.

Context:
{context}

Question: {question}

Options:
{options}

Please think step by step and then provide your answer in the following format:
Answer: [A/B/C/D/E]

Your response:"""
)


# ============================================================
# LLM Generator
# ============================================================


class MedicalLLMGenerator:
    """Medical LLM Generator for answer generation"""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize LLM generator"""
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key
        self.api_key = api_key or self._get_api_key(provider)
        if not self.api_key:
            raise ValueError(
                f"API key not found for {provider}. "
                f"Please set {provider.upper()}_API_KEY environment variable."
            )

        # Get base URL
        self.base_url = base_url or self._get_base_url(provider)

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self.prompt = MEDICAL_RAG_PROMPT

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment"""
        provider_keys = {
            "openai": "OPENAI_API_KEY",
            "zhipu": "ZHIPU_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "moonshot": "MOONSHOT_API_KEY",
        }
        env_var = provider_keys.get(provider.lower(), "OPENAI_API_KEY")
        return os.getenv(env_var)

    def _get_base_url(self, provider: str) -> str:
        """Get base URL for provider"""
        base_urls = {
            "openai": "https://api.openai.com/v1",
            "zhipu": "https://open.bigmodel.cn/api/paas/v4",
            "deepseek": "https://api.deepseek.com/v1",
            "moonshot": "https://api.moonshot.cn/v1",
        }
        return base_urls.get(provider.lower(), "https://api.openai.com/v1")

    def generate(
        self,
        question: str,
        contexts: List[str],
        options: Optional[List[str]] = None,
    ) -> str:
        """
        Generate answer using LLM.

        Args:
            question: The question to answer
            contexts: List of context strings
            options: Optional list of answer options

        Returns:
            Generated answer string
        """
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
        prompt = self.prompt.format(
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


def split_dataset(
    questions: List[Dict],
    dev_size: int = 300,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into development and test sets.

    Args:
        questions: List of questions
        dev_size: Number of questions for development set

    Returns:
        (dev_set, test_set)
    """
    print(f"\nSplitting dataset...")
    print(f"  Development set: {dev_size} questions")
    print(f"  Test set: {len(questions) - dev_size} questions")

    dev_set = questions[:dev_size]
    test_set = questions[dev_size:]

    return dev_set, test_set


def evaluate_single_question(
    vectorstore,
    llm_generator: MedicalLLMGenerator,
    question: str,
    options: List[str],
    correct_answer: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate a single question with full RAG pipeline.

    Flow:
    1. Retrieve documents
    2. Generate answer with LLM
    3. Compare with correct answer

    Returns:
        Dictionary with evaluation results
    """
    # Step 1: Retrieve
    results = vectorstore.similarity_search_with_score(question, k=top_k)
    docs = [doc for doc, score in results]
    scores = [float(score) for doc, score in results]
    contexts = [doc.page_content for doc in docs]

    # Step 2: Generate
    response = llm_generator.generate(question, contexts, options)
    predicted_answer = llm_generator.extract_answer(response)

    # Step 3: Compare
    is_correct = predicted_answer == correct_answer.upper()

    return {
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "response": response,
        "retrieved_docs": len(docs),
        "scores": scores,
        "contexts": contexts,
    }


def evaluate_dataset(
    vectorstore,
    llm_generator: MedicalLLMGenerator,
    questions: List[Dict],
    top_k: int = 5,
    dataset_name: str = "Dataset",
) -> Dict[str, Any]:
    """
    Evaluate on a dataset.

    Returns:
        Evaluation results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating {dataset_name} (top-k={top_k})")
    print(f"{'=' * 60}")

    start_time = time.time()
    results = []
    correct = 0
    total = 0

    for i, q in enumerate(questions, 1):
        question_text = q.get("question", "")
        options = q.get("options", [])
        correct_answer = q.get("answer", "")
        answer_index = q.get("answer_index", -1)

        # Convert answer_index to letter (A=0, B=1, C=2, ...)
        if answer_index >= 0:
            correct_answer_letter = chr(65 + answer_index)  # 65 is ASCII for 'A'
        else:
            correct_answer_letter = correct_answer  # Fallback to original

        try:
            result = evaluate_single_question(
                vectorstore,
                llm_generator,
                question_text,
                options,
                correct_answer_letter,
                top_k,
            )
            results.append(result)

            if result["is_correct"]:
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

        except Exception as e:
            print(f"  ERROR on question {i}: {e}")
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


def find_best_top_k(
    vectorstore,
    llm_generator: MedicalLLMGenerator,
    dev_set: List[Dict],
    k_values: List[int],
) -> Tuple[int, Dict[int, float]]:
    """
    Find the best top-k on development set.

    Returns:
        (best_k, accuracy_dict)
    """
    print(f"\n{'=' * 60}")
    print("Hyperparameter Search: Finding Best top-k")
    print(f"{'=' * 60}")

    accuracy_scores = {}

    for k in k_values:
        results = evaluate_dataset(
            vectorstore,
            llm_generator,
            dev_set,
            top_k=k,
            dataset_name=f"Development Set",
        )
        accuracy_scores[k] = results["accuracy"]
        print(
            f"\n  top-k={k}: Accuracy = {results['accuracy']:.4f} "
            f"({results['correct']}/{results['total_questions']})"
        )

    # Find best k
    best_k = max(accuracy_scores, key=accuracy_scores.get)
    print(f"\n[OK] Best top-k: {best_k} (Accuracy: {accuracy_scores[best_k]:.4f})")

    return best_k, accuracy_scores


def calculate_retrieval_recall_at_k(
    vectorstore,
    questions: List[Dict],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[int, float]:
    """
    Calculate Recall@k for retrieval quality.

    This checks if the correct answer text appears in the retrieved documents.

    Returns:
        Dictionary of k -> recall score
    """
    print(f"\n{'=' * 60}")
    print("Calculating Retrieval Recall@k")
    print(f"{'=' * 60}")

    recall_scores = {}

    for k in k_values:
        print(f"\nEvaluating k={k}...")

        correct = 0
        total = 0

        for i, q in enumerate(questions, 1):
            question_text = q.get("question", "")
            correct_answer = q.get("answer", "")

            try:
                results = vectorstore.similarity_search_with_score(question_text, k=k)
                docs = [doc for doc, score in results]

                # Check if correct answer is in retrieved docs
                found = False
                for doc in docs:
                    content = doc.page_content.lower()
                    if correct_answer.lower() in content:
                        found = True
                        break

                if found:
                    correct += 1
                total += 1

                if i % 50 == 0:
                    print(f"  Processed {i}/{len(questions)} questions...")

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        recall = correct / total if total > 0 else 0
        recall_scores[k] = recall

        print(f"  Recall@{k} = {recall:.4f} ({correct}/{total})")

    return recall_scores


# ============================================================
# Main Evaluation Pipeline
# ============================================================


def main():
    """Main evaluation function"""
    print("=" * 60)
    print("Medical RAG System - Complete Evaluation")
    print("=" * 60)

    # Load configuration
    config = EvaluationConfig()

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Load questions
    questions = load_questions(config.QUESTION_FILE)

    if not questions:
        print("\nNo questions loaded. Exiting...")
        return

    # Split dataset
    dev_set, test_set = split_dataset(questions, config.DEV_SET_SIZE)

    # Initialize vector store
    print(f"\nLoading vector store from {config.VECTOR_STORE_PATH}...")

    if not os.path.exists(config.VECTOR_STORE_PATH):
        print(f"ERROR: Vector store not found: {config.VECTOR_STORE_PATH}")
        print("Run build_vector_index.py first")
        return

    # Load embeddings
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Load vector store
    print("Loading FAISS index...")
    vectorstore = MedicalVectorStore(
        embedding_model=embeddings,
        store_type="faiss",
        persist_directory=config.VECTOR_STORE_PATH,
    )
    vectorstore.load(config.VECTOR_STORE_PATH)

    print("[OK] Vector store loaded")

    # Initialize LLM generator
    print("\nInitializing LLM Generator...")
    print(f"  Provider: {config.LLM_PROVIDER}")
    print(f"  Model: {config.LLM_MODEL}")
    print(f"  Temperature: {config.LLM_TEMPERATURE}")
    print(f"  Base URL: {config.LLM_BASE_URL}")

    try:
        llm_generator = MedicalLLMGenerator(
            provider=config.LLM_PROVIDER,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
        )
        print("[OK] LLM Generator initialized")
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nPlease check your API configuration")
        return

    # ============================================================
    # Step 1: Hyperparameter Search on Development Set
    # ============================================================

    best_k, dev_accuracy_scores = find_best_top_k(
        vectorstore,
        llm_generator,
        dev_set,
        config.TOP_K_VALUES,
    )

    # ============================================================
    # Step 2: Final Evaluation on Test Set with Best top-k
    # ============================================================

    print(f"\n{'=' * 60}")
    print(f"Final Evaluation on Test Set (using best top-k={best_k})")
    print(f"{'=' * 60}")

    test_results = evaluate_dataset(
        vectorstore,
        llm_generator,
        test_set,
        top_k=best_k,
        dataset_name="Test Set",
    )

    # ============================================================
    # Step 3: Calculate Retrieval Recall@k
    # ============================================================

    retrieval_recall = calculate_retrieval_recall_at_k(
        vectorstore,
        test_set,
        k_values=[1, 3, 5, 10],
    )

    # ============================================================
    # Step 4: Save Results
    # ============================================================

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save complete results
    results_file = os.path.join(config.OUTPUT_DIR, f"complete_eval_{timestamp}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "dev_set_size": config.DEV_SET_SIZE,
                    "llm_provider": config.LLM_PROVIDER,
                    "llm_model": config.LLM_MODEL,
                    "vector_store": config.VECTOR_STORE_PATH,
                },
                "hyperparameter_search": {
                    "k_values_tested": config.TOP_K_VALUES,
                    "development_set_accuracy": dev_accuracy_scores,
                    "best_k": best_k,
                },
                "test_set_evaluation": test_results,
                "retrieval_recall_at_k": retrieval_recall,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[OK] Results saved to {results_file}")

    # Save summary
    summary_file = os.path.join(
        config.OUTPUT_DIR, f"evaluation_summary_{timestamp}.txt"
    )
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Medical RAG System - Complete Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL}\n")
        f.write(f"Vector Store: {config.VECTOR_STORE_PATH}\n\n")

        f.write("Dataset Split:\n")
        f.write(f"  Development Set: {len(dev_set)} questions\n")
        f.write(f"  Test Set: {len(test_set)} questions\n\n")

        f.write("Hyperparameter Search (Development Set):\n")
        for k, acc in dev_accuracy_scores.items():
            f.write(f"  top-k={k}: Accuracy = {acc:.4f}\n")
        f.write(f"  [OK] Best top-k: {best_k}\n\n")

        f.write("Test Set Evaluation:\n")
        f.write(f"  Total Questions: {test_results['total_questions']}\n")
        f.write(f"  Correct Answers: {test_results['correct']}\n")
        f.write(f"  Accuracy: {test_results['accuracy']:.4f}\n")
        f.write(f"  Time: {test_results['elapsed_time']:.1f}s\n")
        f.write(f"  Speed: {test_results['questions_per_second']:.2f} q/s\n\n")

        f.write("Retrieval Recall@k:\n")
        for k, recall in retrieval_recall.items():
            f.write(f"  R@{k}: {recall:.4f}\n")

    print(f"[OK] Summary saved to {summary_file}")

    # ============================================================
    # Step 5: Print Final Summary
    # ============================================================

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n📊 Final Results:")
    print(f"  Development Set Size: {len(dev_set)} questions")
    print(f"  Test Set Size: {len(test_set)} questions")
    print(f"  Best top-k: {best_k}")
    print(
        f"\n🎯 Test Set Accuracy: {test_results['accuracy']:.4f} "
        f"({test_results['correct']}/{test_results['total_questions']})"
    )
    print(
        f"\n⏱️  Evaluation Time: {test_results['elapsed_time']:.1f}s "
        f"({test_results['questions_per_second']:.2f} questions/second)"
    )
    print(f"\n📈 Retrieval Quality:")
    for k, recall in retrieval_recall.items():
        print(f"  Recall@{k}: {recall:.4f}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
