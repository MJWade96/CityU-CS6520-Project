"""
Evaluate LLM accuracy WITHOUT RAG (direct LLM inference)

This script evaluates the baseline performance of the LLM without any 
retrieval augmentation - using only the model's internal knowledge.

Usage:
    python evaluate_no_rag.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from langchain_openai import ChatOpenAI


# ============================================================
# Configuration
# ============================================================


class EvalConfig:
    """Evaluation configuration"""

    # Get script directory
    SCRIPT_DIR = Path(__file__).parent

    # LLM Configuration (联通云 DeepSeek V3.2)
    LLM_PROVIDER = "deepseek"
    LLM_MODEL = "2656053fa69c4c2d89c5a691d9d737c3"  # DeepSeek V3.2
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 512
    LLM_BASE_URL = "https://wishub-x6.ctyun.cn/v1"  # 联通云 API 端点
    LLM_API_KEY = "6fcecb364d0647d2883e7f1d3f19d5b9"  # 联通云 API Key

    # File paths
    QUESTION_FILE = str(SCRIPT_DIR / "data" / "evaluation" / "medqa.json")
    OUTPUT_DIR = str(SCRIPT_DIR / "results" / "evaluation")


# ============================================================
# Prompt Template (No Context)
# ============================================================

NO_RAG_PROMPT = """You are a medical expert assistant. Answer the following question based on your medical knowledge.

Question: {question}

Options:
{options}

Please think step by step and then provide your answer in the following format:
Answer: [A/B/C/D/E]

Your response:"""


# ============================================================
# LLM Generator (No RAG)
# ============================================================


class DirectLLMGenerator:
    """Direct LLM generator without RAG"""

    def __init__(
        self,
        provider: str = "deepseek",
        model: str = "2656053fa69c4c2d89c5a691d9d737c3",
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
        return base_urls.get(provider.lower(), "https://api.deepseek.com/v1")

    def generate(
        self,
        question: str,
        options: Optional[List[str]] = None,
    ) -> str:
        """
        Generate answer using LLM without any context.

        Args:
            question: The question to answer
            options: Optional list of answer options

        Returns:
            Generated answer string
        """
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
        prompt = NO_RAG_PROMPT.format(
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


def evaluate_without_rag(
    llm_generator: DirectLLMGenerator,
    questions: List[Dict],
    dataset_name: str = "Dataset",
) -> Dict[str, Any]:
    """
    Evaluate LLM without RAG (direct inference).

    Flow:
    1. Send question directly to LLM (no retrieval)
    2. Extract answer from response
    3. Compare with correct answer

    Returns:
        Evaluation results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating WITHOUT RAG - {dataset_name}")
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
            correct_answer_letter = chr(65 + answer_index)
        else:
            correct_answer_letter = correct_answer

        try:
            # Generate answer WITHOUT any retrieved context
            response = llm_generator.generate(question_text, options)
            predicted_answer = llm_generator.extract_answer(response)

            # Compare
            is_correct = predicted_answer == correct_answer_letter.upper()

            result = {
                "question": question_text,
                "options": options,
                "correct_answer": correct_answer_letter,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response": response,
            }
            results.append(result)

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
        "detailed_results": results,
    }


def main():
    """Main evaluation function"""
    print("=" * 60)
    print("Medical RAG System - Baseline Evaluation (WITHOUT RAG)")
    print("=" * 60)

    # Load configuration
    config = EvalConfig()

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Load questions
    questions = load_questions(config.QUESTION_FILE)

    if not questions:
        print("\nNo questions loaded. Exiting...")
        return

    # Initialize LLM generator
    print(f"\nInitializing LLM Generator (No RAG)...")
    print(f"  Provider: {config.LLM_PROVIDER}")
    print(f"  Model: {config.LLM_MODEL}")
    print(f"  Temperature: {config.LLM_TEMPERATURE}")
    print(f"  Base URL: {config.LLM_BASE_URL}")

    try:
        llm_generator = DirectLLMGenerator(
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
    # Evaluate Without RAG
    # ============================================================

    no_rag_results = evaluate_without_rag(
        llm_generator,
        questions,
        dataset_name="Full Dataset (No RAG)",
    )

    # ============================================================
    # Save Results
    # ============================================================

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save complete results
    results_file = os.path.join(
        config.OUTPUT_DIR, f"no_rag_eval_{timestamp}.json"
    )
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "llm_provider": config.LLM_PROVIDER,
                    "llm_model": config.LLM_MODEL,
                    "evaluation_type": "NO_RAG (Direct LLM Inference)",
                },
                "evaluation_results": no_rag_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[OK] Results saved to {results_file}")

    # Save summary
    summary_file = os.path.join(
        config.OUTPUT_DIR, f"no_rag_summary_{timestamp}.txt"
    )
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Medical RAG System - Baseline Evaluation (WITHOUT RAG)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL}\n")
        f.write("Evaluation Type: Direct LLM Inference (No Retrieval)\n\n")

        f.write("Results:\n")
        f.write(f"  Total Questions: {no_rag_results['total_questions']}\n")
        f.write(f"  Correct Answers: {no_rag_results['correct']}\n")
        f.write(f"  Accuracy: {no_rag_results['accuracy']:.4f}\n")
        f.write(f"  Time: {no_rag_results['elapsed_time']:.1f}s\n")
        f.write(f"  Speed: {no_rag_results['questions_per_second']:.2f} q/s\n\n")

    print(f"[OK] Summary saved to {summary_file}")

    # ============================================================
    # Print Final Summary
    # ============================================================

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n📊 Baseline Results (No RAG):")
    print(f"  Total Questions: {no_rag_results['total_questions']}")
    print(f"  Correct Answers: {no_rag_results['correct']}")
    print(
        f"\n🎯 Accuracy: {no_rag_results['accuracy']:.4f} "
        f"({no_rag_results['correct']}/{no_rag_results['total_questions']})"
    )
    print(
        f"\n⏱️  Evaluation Time: {no_rag_results['elapsed_time']:.1f}s "
        f"({no_rag_results['questions_per_second']:.2f} questions/second)"
    )

    print(f"\n{'=' * 60}")
    print("This is the BASELINE performance without RAG.")
    print("Compare with RAG-enhanced results to measure improvement.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
