"""
MedQA Evaluation Module

Implements evaluation for MedQA-US (USMLE) dataset:
1. Accuracy - Standard multiple choice accuracy
2. Recall@k - Whether correct answer is in retrieved top-k documents

Data Format:
- MedQA-US: JSON format with questions, options, and answer
"""

import os
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MedQAQuestion:
    """Single MedQA question"""
    question: str
    options: List[str]
    answer: str
    answer_index: int
    source: str = "medqa"


class MedQAEvaluator:
    """
    MedQA Evaluation for RAG Systems

    Metrics:
    - Accuracy: (correct answers / total questions)
    - Recall@k: Whether retrieved docs contain answer
    """

    def __init__(
        self,
        data_path: str = None,
        dev_size: int = 300,
        test_size: int = 500,
        seed: int = 42
    ):
        """
        Initialize MedQA evaluator.

        Args:
            data_path: Path to MedQA JSON file
            dev_size: Number of questions for dev set
            test_size: Number of questions for test set
            seed: Random seed for splitting
        """
        self.dev_size = dev_size
        self.test_size = test_size
        self.seed = seed

        self.questions: List[MedQAQuestion] = []
        self.dev_questions: List[MedQAQuestion] = []
        self.test_questions: List[MedQAQuestion] = []

        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path: str) -> None:
        """Load MedQA data from JSON file"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"MedQA data not found: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.questions = self._parse_medqa(data)
        self._split_data()

        print(f"Loaded {len(self.questions)} MedQA questions")
        print(f"  Dev set: {len(self.dev_questions)} questions")
        print(f"  Test set: {len(self.test_questions)} questions")

    def _parse_medqa(self, data: List[Dict]) -> List[MedQAQuestion]:
        """Parse MedQA data format"""
        questions = []

        for item in data:
            question_text = item.get('question', '')

            options = item.get('options', {})
            if isinstance(options, dict):
                option_list = [options[k] for k in sorted(options.keys())]
            else:
                option_list = options if isinstance(options, list) else []

            answer = item.get('answer', '')
            answer_index = item.get('answer_index', -1)

            if answer and answer in option_list:
                answer_index = option_list.index(answer)

            questions.append(MedQAQuestion(
                question=question_text,
                options=option_list,
                answer=answer,
                answer_index=answer_index,
                source=item.get('source', 'medqa')
            ))

        return questions

    def _split_data(self) -> None:
        """Split questions into dev and test sets"""
        random.seed(self.seed)
        shuffled = self.questions.copy()
        random.shuffle(shuffled)

        self.dev_questions = shuffled[:self.dev_size]
        self.test_questions = shuffled[self.dev_size:self.dev_size + self.test_size]

    def get_dev_set(self) -> List[MedQAQuestion]:
        """Get development set"""
        return self.dev_questions

    def get_test_set(self) -> List[MedQAQuestion]:
        """Get test set"""
        return self.test_questions

    def evaluate_accuracy(
        self,
        predictions: List[str],
        questions: List[MedQAQuestion] = None
    ) -> Dict[str, float]:
        """
        Calculate accuracy on predictions.

        Args:
            predictions: List of predicted answers
            questions: List of questions (uses test set if None)

        Returns:
            Dictionary with accuracy score
        """
        if questions is None:
            questions = self.test_questions

        if len(predictions) != len(questions):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) "
                f"doesn't match questions ({len(questions)})"
            )

        correct = 0
        for pred, q in zip(predictions, questions):
            if pred.lower().strip() == q.answer.lower().strip():
                correct += 1

        accuracy = correct / len(questions) if questions else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(questions)
        }

    def evaluate_recall_at_k(
        self,
        retrieved_contexts: List[List[str]],
        questions: List[MedQAQuestion] = None,
        k: int = 5
    ) -> Dict[str, float]:
        """
        Calculate Recall@k - whether retrieved docs contain answer information.

        Args:
            retrieved_contexts: List of retrieved contexts for each question
            questions: List of questions (uses test set if None)
            k: Top-k documents to check

        Returns:
            Dictionary with recall@k score
        """
        if questions is None:
            questions = self.test_questions

        if len(retrieved_contexts) != len(questions):
            raise ValueError(
                f"Number of contexts ({len(retrieved_contexts)}) "
                f"doesn't match questions ({len(questions)})"
            )

        hits = 0
        for contexts, q in zip(retrieved_contexts, questions):
            top_k_contexts = contexts[:k]

            context_text = ' '.join(top_k_contexts).lower()

            answer_in_context = q.answer.lower() in context_text
            for option in q.options:
                if option.lower() in context_text:
                    answer_in_context = True
                    break

            if answer_in_context:
                hits += 1

        recall = hits / len(questions) if questions else 0

        return {
            f'recall@{k}': recall,
            'hits': hits,
            'total': len(questions),
            'k': k
        }

    def evaluate(
        self,
        predictions: List[str],
        retrieved_contexts: List[List[str]],
        questions: List[MedQAQuestion] = None,
        k_values: List[int] = None
    ) -> Dict[str, Any]:
        """
        Full evaluation including accuracy and recall@k.

        Args:
            predictions: List of predicted answers
            retrieved_contexts: List of retrieved contexts
            questions: List of questions
            k_values: List of k values for recall (default: [5, 10])

        Returns:
            Complete evaluation results
        """
        if questions is None:
            questions = self.test_questions

        if k_values is None:
            k_values = [5, 10]

        results = {
            'total_questions': len(questions),
        }

        accuracy_result = self.evaluate_accuracy(predictions, questions)
        results['accuracy'] = accuracy_result

        for k in k_values:
            recall_result = self.evaluate_recall_at_k(
                retrieved_contexts, questions, k
            )
            results[f'recall@{k}'] = recall_result[f'recall@{k}']

        return results


def load_medqa_from_huggingface() -> List[Dict]:
    """
    Load MedQA dataset from HuggingFace datasets.

    Returns:
        List of MedQA questions in dict format
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install: pip install datasets")

    print("Loading MedQA dataset from HuggingFace...")
    dataset = load_dataset("bigbio/medqa")

    data = []
    for item in dataset['test']:
        question = item.get('question', '')

        options = item.get('options', {})
        if isinstance(options, dict):
            option_list = [options[k] for k in sorted(options.keys())]
        else:
            option_list = options if isinstance(options, list) else []

        answer = item.get('answer', '')
        answer_key = item.get('answer_key', '')

        if answer_key and isinstance(options, dict):
            answer = options.get(answer_key, answer)

        data.append({
            'question': question,
            'options': option_list,
            'answer': answer,
            'source': 'medqa'
        })

    print(f"Loaded {len(data)} MedQA questions")
    return data


def save_medqa_json(output_path: str, data: List[Dict]) -> None:
    """Save MedQA data to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} questions to {output_path}")


def demo():
    """Demonstrate MedQA evaluation"""
    print("=" * 60)
    print("MedQA Evaluation Demo")
    print("=" * 60)

    sample_data = [
        {
            "question": "A 55-year-old man with hypertension presents with chest pain for 2 hours. ECG shows ST elevation in leads II, III, and aVF. What is the most likely diagnosis?",
            "options": {
                "A": "Acute inferior wall myocardial infarction",
                "B": "Acute anterior wall myocardial infarction",
                "C": "Unstable angina",
                "D": "Aortic dissection"
            },
            "answer": "Acute inferior wall myocardial infarction",
            "answer_index": 0,
            "source": "medqa"
        },
        {
            "question": "What is the first-line medication for type 2 diabetes mellitus?",
            "options": {
                "A": "Metformin",
                "B": "Insulin",
                "C": "Glipizide",
                "D": "Sitagliptin"
            },
            "answer": "Metformin",
            "answer_index": 0,
            "source": "medqa"
        }
    ]

    evaluator = MedQAEvaluator(dev_size=1, test_size=1, seed=42)
    evaluator.questions = [MedQAQuestion(**item) for item in sample_data]
    evaluator._split_data()

    predictions = ["Acute inferior wall myocardial infarction", "Metformin"]

    retrieved_contexts = [
        ["Inferior wall MI shows ST elevation in leads II, III, aVF."],
        ["Metformin is the first-line medication for type 2 diabetes."]
    ]

    results = evaluator.evaluate(predictions, retrieved_contexts)

    print("\nEvaluation Results:")
    print(f"  Accuracy: {results['accuracy']['accuracy']:.2%}")
    print(f"  Recall@5: {results['recall@5']:.2%}")
    print(f"  Recall@10: {results['recall@10']:.2%}")


if __name__ == "__main__":
    demo()
