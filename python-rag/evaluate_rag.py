"""
Medical RAG Evaluation System

Evaluates the RAG system on MedQA-US (USMLE) questions.
Calculates:
1. Accuracy - Primary metric
2. Recall@k - Retrieval quality metric

Usage:
    python evaluate_rag.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.rag.retriever import MedicalRetriever
from app.rag.generator import MedicalGenerator


class RAGEvaluator:
    """Evaluator for Medical RAG System"""
    
    def __init__(
        self,
        retriever: MedicalRetriever = None,
        generator: MedicalGenerator = None,
        top_k: int = 5
    ):
        """
        Initialize evaluator
        
        Args:
            retriever: Medical retriever instance
            generator: Medical generator instance (optional for retrieval-only eval)
            top_k: Number of documents to retrieve
        """
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        
        self.correct_retrieval = 0
        self.total_questions = 0
        self.results = []
    
    def load_questions(self, question_file: str) -> List[Dict]:
        """
        Load MedQA questions
        
        Args:
            question_file: Path to JSON file with questions
        
        Returns:
            List of question dictionaries
        """
        print(f"Loading questions from {question_file}...")
        
        with open(question_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
        
        print(f"✓ Loaded {len(questions)} questions")
        return questions
    
    def evaluate_retrieval(
        self,
        question: str,
        correct_answer: str,
        question_id: str = ""
    ) -> Tuple[bool, List[Dict], List[float]]:
        """
        Evaluate retrieval for a single question
        
        Args:
            question: Question text
            correct_answer: Correct answer text
            question_id: Question identifier
        
        Returns:
            Tuple of (retrieval_successful, retrieved_docs, scores)
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        
        # Retrieve documents
        results = self.retriever.similarity_search_with_score(
            question,
            k=self.top_k
        )
        
        docs = [doc for doc, score in results]
        scores = [float(score) for doc, score in results]
        
        # Check if any retrieved doc contains the correct answer
        retrieval_success = False
        for doc in docs:
            content = doc.page_content.lower()
            if correct_answer.lower() in content:
                retrieval_success = True
                break
        
        return retrieval_success, docs, scores
    
    def evaluate_qa(
        self,
        question: str,
        options: Dict[str, str],
        correct_answer: str,
        question_id: str = ""
    ) -> Tuple[str, str, List[Dict]]:
        """
        Evaluate full QA pipeline for a single question
        
        Args:
            question: Question text
            options: Dictionary of answer options
            correct_answer: Correct answer text
            question_id: Question identifier
        
        Returns:
            Tuple of (predicted_answer, correct_answer, retrieved_docs)
        """
        if not self.retriever or not self.generator:
            raise ValueError("Retriever or Generator not initialized")
        
        # Retrieve documents
        docs = self.retriever.similarity_search(question, k=self.top_k)
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer
        answer = self.generator.generate_answer(
            question=question,
            options=options,
            context=context
        )
        
        return answer, correct_answer, docs
    
    def run_evaluation(
        self,
        questions: List[Dict],
        evaluation_type: str = "retrieval"
    ) -> Dict:
        """
        Run evaluation on all questions
        
        Args:
            questions: List of question dictionaries
            evaluation_type: "retrieval" or "qa"
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Running {evaluation_type.upper()} Evaluation")
        print(f"{'=' * 60}")
        print(f"Total questions: {len(questions)}")
        print(f"Top-k: {self.top_k}")
        
        start_time = time.time()
        
        correct = 0
        total = 0
        results = []
        
        for i, q in enumerate(questions, 1):
            question_id = q.get("id", f"q_{i}")
            question_text = q.get("question", "")
            correct_answer = q.get("answer", "")
            options = q.get("options", {})
            
            # Progress
            if i % 10 == 0 or i == len(questions):
                elapsed = time.time() - start_time
                qps = i / elapsed if elapsed > 0 else 0
                print(f"  Question {i}/{len(questions)} ({elapsed:.1f}s, {qps:.2f} q/s)")
            
            try:
                if evaluation_type == "retrieval":
                    # Evaluate retrieval only
                    success, docs, scores = self.evaluate_retrieval(
                        question_text,
                        correct_answer,
                        question_id
                    )
                    
                    if success:
                        correct += 1
                    
                    result = {
                        "id": question_id,
                        "question": question_text[:100],
                        "correct_answer": correct_answer,
                        "retrieval_success": success,
                        "retrieved_docs": len(docs),
                        "scores": scores,
                    }
                
                elif evaluation_type == "qa":
                    # Evaluate full QA
                    predicted, correct_ans, docs = self.evaluate_qa(
                        question_text,
                        options,
                        correct_answer,
                        question_id
                    )
                    
                    is_correct = predicted.lower() == correct_answer.lower()
                    if is_correct:
                        correct += 1
                    
                    result = {
                        "id": question_id,
                        "question": question_text[:100],
                        "options": options,
                        "predicted_answer": predicted,
                        "correct_answer": correct_ans,
                        "is_correct": is_correct,
                    }
                
                results.append(result)
                total += 1
            
            except Exception as e:
                print(f"  ERROR on question {i}: {e}")
                continue
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        
        metrics = {
            "total_questions": total,
            "correct": correct,
            "accuracy": accuracy,
            "elapsed_time": elapsed,
            "questions_per_second": total / elapsed if elapsed > 0 else 0,
            "top_k": self.top_k,
            "evaluation_type": evaluation_type,
        }
        
        return {
            "metrics": metrics,
            "results": results
        }
    
    def calculate_recall_at_k(
        self,
        questions: List[Dict],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[int, float]:
        """
        Calculate Recall@k for different k values
        
        Args:
            questions: List of question dictionaries
            k_values: List of k values to evaluate
        
        Returns:
            Dictionary mapping k to Recall@k score
        """
        print(f"\n{'=' * 60}")
        print("Calculating Recall@k")
        print(f"{'=' * 60}")
        
        recall_scores = {}
        
        for k in k_values:
            print(f"\nEvaluating k={k}...")
            
            self.top_k = k
            correct = 0
            total = 0
            
            for i, q in enumerate(questions, 1):
                question_text = q.get("question", "")
                correct_answer = q.get("answer", "")
                
                try:
                    success, _, _ = self.evaluate_retrieval(
                        question_text,
                        correct_answer
                    )
                    
                    if success:
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


def main():
    """Main evaluation function"""
    print("=" * 60)
    print("Medical RAG System Evaluation")
    print("=" * 60)
    
    # Configuration
    question_file = "./data/evaluation/medqa_test.json"
    output_dir = "./results/evaluation"
    top_k = 5
    
    # Parse arguments
    if "--questions" in sys.argv:
        idx = sys.argv.index("--questions") + 1
        if idx < len(sys.argv):
            question_file = sys.argv[idx]
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output") + 1
        if idx < len(sys.argv):
            output_dir = sys.argv[idx]
    
    if "--top-k" in sys.argv:
        idx = sys.argv.index("--top-k") + 1
        if idx < len(sys.argv):
            top_k = int(sys.argv[idx])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load questions
    if not os.path.exists(question_file):
        print(f"ERROR: Question file not found: {question_file}")
        print("Make sure to download and process MedQA dataset first")
        return
    
    # Initialize retriever
    print("\nInitializing Medical RAG System...")
    retriever = MedicalRetriever(
        vector_store_path="./data/vector_store/faiss_index",
        top_k=top_k
    )
    print("✓ Retriever initialized")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(
        retriever=retriever,
        top_k=top_k
    )
    
    # Load questions
    questions = evaluator.load_questions(question_file)
    
    # Run retrieval evaluation
    eval_results = evaluator.run_evaluation(
        questions[:100],  # Evaluate on first 100 questions for speed
        evaluation_type="retrieval"
    )
    
    # Calculate Recall@k
    recall_scores = evaluator.calculate_recall_at_k(
        questions[:100],
        k_values=[1, 3, 5, 10]
    )
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Evaluation Summary")
    print(f"{'=' * 60}")
    
    metrics = eval_results["metrics"]
    print(f"Total questions: {metrics['total_questions']}")
    print(f"Retrieval Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total_questions']})")
    print(f"Time: {metrics['elapsed_time']:.1f}s ({metrics['questions_per_second']:.2f} q/s)")
    
    print(f"\nRecall@k:")
    for k, score in sorted(recall_scores.items()):
        print(f"  R@{k}: {score:.4f}")
    
    # Save results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "recall_at_k": recall_scores,
            "detailed_results": eval_results["results"]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Medical RAG System Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Questions evaluated: {metrics['total_questions']}\n")
        f.write(f"Top-k: {top_k}\n\n")
        f.write("Metrics:\n")
        f.write(f"  Retrieval Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  Time: {metrics['elapsed_time']:.1f}s\n\n")
        f.write("Recall@k:\n")
        for k, score in sorted(recall_scores.items()):
            f.write(f"  R@{k}: {score:.4f}\n")
    
    print(f"✓ Summary saved to {summary_file}")
    
    print(f"\n{'=' * 60}")
    print("Evaluation Complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
