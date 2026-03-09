"""
Medical RAG Prompt Templates

Standard RAG prompt templates following the规划的 template:
"Given the following context: {retrieved_contexts}. Answer the question: {question}. Choose from A, B, C, D."
"""

from typing import List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


MEDICAL_RAG_PROMPT = """Given the following context from medical knowledge base, answer the question.

Context:
{context}

Question: {question}

Options: {options}

Instructions:
- Base your answer ONLY on the provided context
- If the context doesn't contain enough information to answer, say "I cannot determine the answer based on the provided information"
- Choose from the given options (A, B, C, or D)

Answer:"""

SIMPLE_RAG_PROMPT = """Given the following context, answer the question.

Context:
{context}

Question: {question}

Answer the question based on the context above. If the context doesn't contain enough information, say so."""


CONTEXT_QA_PROMPT = """You are a medical AI assistant. Use the following context to answer the medical question.

Context:
{context}

Question: {question}

{options}

Select the best answer based on the context and your medical knowledge."""


SYSTEM_MESSAGE = """You are a helpful medical AI assistant. Your role is to answer medical questions based on the provided context from a medical knowledge base. 

Important guidelines:
1. Only use information from the provided context
2. If insufficient information is provided, acknowledge that
3. Provide accurate medical information
4. Do not make up or hallucinate medical facts
5. For multiple choice questions, select the most appropriate option"""


def create_rag_prompt_template(
    template_type: str = "medical",
    include_options: bool = True
) -> PromptTemplate:
    """
    Create a RAG prompt template.

    Args:
        template_type: Type of template ('medical', 'simple', 'context_qa')
        include_options: Whether to include options field

    Returns:
        PromptTemplate instance
    """
    if template_type == "medical":
        if include_options:
            return PromptTemplate(
                template=MEDICAL_RAG_PROMPT,
                input_variables=["context", "question", "options"]
            )
        else:
            return PromptTemplate(
                template=SIMPLE_RAG_PROMPT,
                input_variables=["context", "question"]
            )
    elif template_type == "context_qa":
        return PromptTemplate(
            template=CONTEXT_QA_PROMPT,
            input_variables=["context", "question", "options"]
        )
    else:
        return PromptTemplate(
            template=SIMPLE_RAG_PROMPT,
            input_variables=["context", "question"]
        )


def create_chat_prompt_template(
    system_message: str = None,
    template_type: str = "medical"
) -> ChatPromptTemplate:
    """
    Create a ChatPromptTemplate for LLM-based generation.

    Args:
        system_message: Custom system message
        template_type: Type of human message template

    Returns:
        ChatPromptTemplate instance
    """
    if system_message is None:
        system_message = SYSTEM_MESSAGE

    if template_type == "medical":
        human_template = """Context:
{context}

Question: {question}

Options: {options}

Answer:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    else:
        human_template = """Context:
{context}

Question: {question}

Answer:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(human_template)
        ])


def format_context(documents: List[str], max_docs: int = 5) -> str:
    """
    Format retrieved documents into context string.

    Args:
        documents: List of document contents
        max_docs: Maximum number of documents to include

    Returns:
        Formatted context string
    """
    context_parts = []
    for i, doc in enumerate(documents[:max_docs]):
        context_parts.append(f"[{i+1}] {doc}")

    return "\n\n".join(context_parts)


def format_options(options: List[str]) -> str:
    """
    Format multiple choice options.

    Args:
        options: List of option strings

    Returns:
        Formatted options string
    """
    option_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]

    formatted = []
    for i, option in enumerate(options):
        letter = option_letters[i] if i < len(option_letters) else str(i)
        formatted.append(f"{letter}. {option}")

    return "\n".join(formatted)


def create_rag_inputs(
    retrieved_docs: List[str],
    question: str,
    options: List[str] = None,
    max_docs: int = 5
) -> Dict[str, Any]:
    """
    Create inputs for RAG prompt.

    Args:
        retrieved_docs: Retrieved document contents
        question: Question string
        options: List of options (for multiple choice)
        max_docs: Maximum docs to include

    Returns:
        Dictionary of prompt inputs
    """
    inputs = {
        "context": format_context(retrieved_docs, max_docs),
        "question": question
    }

    if options:
        inputs["options"] = format_options(options)

    return inputs


class MedicalPromptManager:
    """Manager for medical RAG prompts"""

    def __init__(self, template_type: str = "medical"):
        self.template_type = template_type
        self.prompt_template = create_rag_prompt_template(template_type)
        self.chat_prompt = create_chat_prompt_template(template_type)

    def format_prompt(
        self,
        retrieved_docs: List[str],
        question: str,
        options: List[str] = None
    ) -> str:
        """Format prompt with retrieved context and question"""
        inputs = create_rag_inputs(retrieved_docs, question, options)
        return self.prompt_template.format(**inputs)

    def get_langchain_prompt(self, options: List[str] = None):
        """Get LangChain prompt template"""
        if options:
            return create_rag_prompt_template(self.template_type, include_options=True)
        else:
            return create_rag_prompt_template(self.template_type, include_options=False)


def demo():
    """Demonstrate prompt templates"""
    print("=" * 60)
    print("Medical RAG Prompt Templates Demo")
    print("=" * 60)

    context = [
        "Hypertension is defined as systolic blood pressure >= 130 mmHg or diastolic >= 80 mmHg.",
        "First-line treatments for hypertension include thiazide diuretics, ACE inhibitors, ARBs, and calcium channel blockers."
    ]

    question = "What is the definition of hypertension?"
    options = ["SBP >= 120 mmHg", "SBP >= 130 mmHg", "SBP >= 140 mmHg", "SBP >= 150 mmHg"]

    print("\n--- Simple Prompt ---")
    manager = MedicalPromptManager("simple")
    print(manager.format_prompt(context, question))

    print("\n--- Medical Prompt with Options ---")
    manager = MedicalPromptManager("medical")
    print(manager.format_prompt(context, question, options))


if __name__ == "__main__":
    demo()
