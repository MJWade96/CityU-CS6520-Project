"""
Medical RAG System - API Key Version
LLM only used in Generator Module (via API)

Architecture:
- Embedding: sentence-transformers (local, no LLM)
- Vector Store: FAISS (local)
- Retrieval: Dense retrieval (no LLM)
- Generator: LLM API (OpenAI / Zhipu / DeepSeek)
- Evaluation: Rule-based + Statistical (no LLM)
"""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import json

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Embeddings - Local model, no LLM
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector store - Local
from langchain_community.vectorstores import FAISS

# Text processing - No LLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LLM - API based
from langchain_openai import ChatOpenAI

# Document loader - Real data from external file
from app.rag.document_loader import load_medical_knowledge_base
from app.rag.data_paths import VECTOR_STORE_DIR


@dataclass
class MedicalRAGConfig:
    """Configuration for Medical RAG System - API Key Version"""

    # LLM Settings (Only used in Generator)
    llm_provider: str = "openai"  # openai, zhipu, deepseek
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_api_key: str = ""  # Set via environment variable or config
    llm_base_url: str = ""  # Optional: for custom endpoints

    # Embedding Settings (Local, no LLM)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # or "cuda"

    # Retrieval Settings (No LLM)
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5

    # Vector Store
    vector_store_path: str = str(VECTOR_STORE_DIR)


class APIGenerator:
    """
    API-based Generator Module

    The ONLY module that uses LLM (via API).
    Supports OpenAI, Zhipu AI, DeepSeek, etc.
    """

    # Provider configurations
    PROVIDERS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        },
        "zhipu": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "default_model": "glm-4-flash",
            "models": ["glm-4", "glm-4-flash", "glm-4-plus"],
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek-chat",
            "models": ["deepseek-chat", "deepseek-coder"],
        },
        "moonshot": {
            "base_url": "https://api.moonshot.cn/v1",
            "default_model": "moonshot-v1-8k",
            "models": ["moonshot-v1-8k", "moonshot-v1-32k"],
        },
    }

    MEDICAL_PROMPT = PromptTemplate.from_template(
        """你是一个专业的医疗诊断助手。请基于提供的医学文献上下文回答问题。

重要指导原则：
1. 回答必须主要基于提供的上下文内容
2. 如果上下文信息不足，请明确说明
3. 引用具体的上下文内容作为依据
4. 使用专业但易懂的医学术语
5. 始终建议咨询医疗专业人员

医学文献上下文：
{context}

问题：{question}

请提供清晰、基于证据的回答："""
    )

    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        """
        Initialize API-based generator.

        Args:
            provider: API provider (openai, zhipu, deepseek, moonshot)
            model: Model name (auto-selected if not specified)
            api_key: API key (or set via environment variable)
            base_url: Custom API endpoint (optional)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get provider config
        provider_config = self.PROVIDERS.get(provider, self.PROVIDERS["openai"])

        # Set model
        self.model = model or provider_config["default_model"]

        # Set API key
        self.api_key = api_key or self._get_api_key(provider)
        if not self.api_key:
            raise ValueError(
                f"API key not found for {provider}. "
                f"Please set {provider.upper()}_API_KEY environment variable "
                f"or pass api_key parameter."
            )

        # Set base URL
        self.base_url = base_url or provider_config["base_url"]

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        print(f"Initialized {provider} generator")
        print(f"  Model: {self.model}")
        print(f"  Base URL: {self.base_url}")

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment variables"""
        env_vars = {
            "openai": ["OPENAI_API_KEY"],
            "zhipu": ["ZHIPU_API_KEY", "ZHIPUAI_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY"],
            "moonshot": ["MOONSHOT_API_KEY"],
        }

        for var in env_vars.get(provider, []):
            key = os.getenv(var)
            if key:
                return key

        return None

    def generate(
        self, question: str, contexts: List[str], sources: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate answer using API LLM.

        Args:
            question: User question
            contexts: Retrieved context strings
            sources: Optional source metadata

        Returns:
            Generated answer string
        """
        # Format context
        formatted_context = self._format_context(contexts, sources)

        # Build prompt
        prompt = self.MEDICAL_PROMPT.format(
            context=formatted_context, question=question
        )

        # Generate response
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

    def _format_context(
        self, contexts: List[str], sources: Optional[List[Dict]] = None
    ) -> str:
        """Format contexts for prompt"""
        if not contexts:
            return "未找到相关医学文献。"

        formatted_parts = []
        for i, ctx in enumerate(contexts, 1):
            if sources and i <= len(sources):
                source_info = sources[i - 1]
                formatted_parts.append(
                    f"[来源 {i}: {source_info.get('source', '未知')}]\n" f"{ctx}"
                )
            else:
                formatted_parts.append(f"[文档 {i}]\n{ctx}")

        return "\n\n---\n\n".join(formatted_parts)


class RuleBasedEvaluator:
    """
    Rule-Based Evaluator - No LLM Required

    Uses statistical and rule-based methods for evaluation.
    """

    def __init__(self):
        """Initialize rule-based evaluator"""
        pass

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Document],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality using statistical methods.
        """
        metrics = {}

        # Keyword overlap
        query_words = set(query.lower().split())
        overlap_scores = []
        for doc in retrieved_docs:
            doc_words = set(doc.page_content.lower().split())
            overlap = (
                len(query_words & doc_words) / len(query_words) if query_words else 0
            )
            overlap_scores.append(overlap)

        metrics["keyword_overlap"] = (
            sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0
        )

        # Length ratio
        total_length = sum(len(doc.page_content) for doc in retrieved_docs)
        metrics["avg_content_length"] = (
            total_length / len(retrieved_docs) if retrieved_docs else 0
        )

        # Diversity
        all_content = " ".join(doc.page_content for doc in retrieved_docs)
        unique_words = len(set(all_content.lower().split()))
        total_words = len(all_content.split())
        metrics["content_diversity"] = (
            unique_words / total_words if total_words > 0 else 0
        )

        return metrics

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate answer quality using rule-based methods.
        """
        metrics = {}

        # Context coverage
        answer_words = set(answer.lower().split())
        context_words = set(" ".join(contexts).lower().split())
        coverage = (
            len(answer_words & context_words) / len(answer_words) if answer_words else 0
        )
        metrics["context_coverage"] = coverage

        # Answer length score
        answer_len = len(answer.split())
        if answer_len < 20:
            metrics["length_score"] = 0.3
        elif answer_len > 500:
            metrics["length_score"] = 0.7
        else:
            metrics["length_score"] = 1.0

        # Question relevance
        question_words = set(question.lower().split())
        question_in_answer = (
            len(question_words & answer_words) / len(question_words)
            if question_words
            else 0
        )
        metrics["question_relevance"] = question_in_answer

        # Ground truth comparison
        if ground_truth:
            gt_words = set(ground_truth.lower().split())
            gt_overlap = len(gt_words & answer_words) / len(gt_words) if gt_words else 0
            metrics["ground_truth_overlap"] = gt_overlap

        return metrics

    def check_safety(self, answer: str) -> Dict[str, Any]:
        """
        Check answer safety using rules.
        """
        answer_lower = answer.lower()

        # Dangerous patterns
        dangerous_patterns = [
            "停止服药",
            "不要看医生",
            "一定可以治愈",
            "绝对安全",
            "没有任何副作用",
            "替代医生",
        ]
        found_dangerous = [p for p in dangerous_patterns if p in answer_lower]

        # Required disclaimers
        disclaimer_patterns = ["咨询", "医生", "医疗专业人员", "仅供参考"]
        found_disclaimers = [p for p in disclaimer_patterns if p in answer_lower]

        # Calculate safety score
        safety_score = 1.0
        if found_dangerous:
            safety_score -= 0.3 * len(found_dangerous)
        if not found_disclaimers:
            safety_score -= 0.2

        safety_score = max(0.0, min(1.0, safety_score))

        return {
            "safety_score": safety_score,
            "dangerous_patterns_found": found_dangerous,
            "disclaimers_found": found_disclaimers,
            "is_safe": safety_score >= 0.7,
        }

    def comprehensive_evaluation(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        retrieved_docs: List[Document],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation without LLM.
        """
        retrieval_metrics = self.evaluate_retrieval(
            question, retrieved_docs, ground_truth
        )
        answer_metrics = self.evaluate_answer(question, answer, contexts, ground_truth)
        safety_metrics = self.check_safety(answer)

        # Calculate overall score
        scores = [
            retrieval_metrics.get("keyword_overlap", 0),
            answer_metrics.get("context_coverage", 0),
            answer_metrics.get("question_relevance", 0),
            safety_metrics["safety_score"],
        ]
        overall_score = sum(scores) / len(scores)

        return {
            "overall_score": overall_score,
            "retrieval": retrieval_metrics,
            "answer": answer_metrics,
            "safety": safety_metrics,
        }


class MedicalRAGSystem:
    """
    Complete Medical RAG System

    Architecture:
    - Embedding: Local sentence-transformers (NO LLM)
    - Vector Store: FAISS (NO LLM)
    - Retrieval: Dense retrieval (NO LLM)
    - Generator: API LLM (ONLY module with LLM)
    - Evaluation: Rule-based (NO LLM)
    """

    def __init__(self, config: Optional[MedicalRAGConfig] = None):
        """Initialize the RAG system"""
        self.config = config or MedicalRAGConfig()

        # Initialize embedding model (NO LLM)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={"device": self.config.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize vector store (NO LLM)
        self.vector_store = None

        # Initialize generator (ONLY module with LLM via API)
        self.generator = APIGenerator(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )

        # Initialize evaluator (NO LLM)
        self.evaluator = RuleBasedEvaluator()

        # Document storage
        self.documents: List[Document] = []

        self.is_initialized = False

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the knowledge base"""
        self.documents.extend(documents)

    def build_index(self) -> None:
        """Build vector index from documents"""
        if not self.documents:
            raise ValueError("No documents to index")

        # Split documents (NO LLM)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        split_docs = text_splitter.split_documents(self.documents)

        # Build vector store (NO LLM)
        self.vector_store = FAISS.from_documents(split_docs, self.embedding_model)

        print(f"Built index with {len(split_docs)} chunks")

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant documents (NO LLM)"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        k = k or self.config.top_k
        return self.vector_store.similarity_search(query, k=k)

    def generate(self, question: str, documents: List[Document]) -> str:
        """Generate answer using API LLM (ONLY LLM usage)"""
        contexts = [doc.page_content for doc in documents]
        sources = [doc.metadata for doc in documents]

        return self.generator.generate(question, contexts, sources)

    def query(self, question: str, evaluate: bool = True) -> Dict[str, Any]:
        """
        Process a medical query.

        Flow:
        1. Retrieve documents (NO LLM)
        2. Generate answer with API LLM (LLM)
        3. Evaluate results (NO LLM)
        """
        if not self.is_initialized:
            raise ValueError("System not initialized. Call initialize() first.")

        # Step 1: Retrieve (NO LLM)
        documents = self.retrieve(question)

        # Step 2: Generate (LLM via API)
        answer = self.generate(question, documents)

        # Step 3: Evaluate (NO LLM)
        if evaluate:
            contexts = [doc.page_content for doc in documents]
            eval_results = self.evaluator.comprehensive_evaluation(
                question=question,
                answer=answer,
                contexts=contexts,
                retrieved_docs=documents,
            )
        else:
            eval_results = None

        # Format sources
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title", "Unknown"),
                "content": doc.page_content[:200] + "...",
            }
            for doc in documents
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "evaluation": eval_results,
        }

    def initialize(self, use_real_data: bool = True) -> None:
        """Initialize the system"""
        if use_real_data:
            self._load_real_knowledge()
        else:
            self._load_sample_knowledge()

        if self.documents:
            self.build_index()

        self.is_initialized = True
        print("Medical RAG System initialized successfully")
        print(f"  - Embedding: {self.config.embedding_model} (Local, NO LLM)")
        print(f"  - LLM: {self.config.llm_provider}/{self.config.llm_model} (API)")
        print(f"  - Evaluation: Rule-based (NO LLM)")

    def _load_real_knowledge(self) -> None:
        """Load real medical knowledge from external JSON file"""
        try:
            kb = load_medical_knowledge_base()
            docs = kb.get_documents()
            for doc in docs:
                self.add_documents([doc])
            print(f"Loaded {len(docs)} real medical documents from external file")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Falling back to sample data...")
            self._load_sample_knowledge()

    def _load_sample_knowledge(self) -> None:
        """Load sample medical knowledge"""
        sample_docs = [
            Document(
                page_content="""高血压临床实践指南

血压分类：
- 正常血压：收缩压 < 120 mmHg 且 舒张压 < 80 mmHg
- 血压升高：收缩压 120-129 mmHg 且 舒张压 < 80 mmHg
- 1级高血压：收缩压 130-139 mmHg 或 舒张压 80-89 mmHg
- 2级高血压：收缩压 ≥ 140 mmHg 或 舒张压 ≥ 90 mmHg

治疗方案：
一线药物包括：
- 噻嗪类利尿剂
- 钙通道阻滞剂
- ACE抑制剂或ARB类药物

目标血压：大多数成人 < 130/80 mmHg""",
                metadata={
                    "source": "AHA/ACC指南",
                    "title": "高血压管理指南",
                    "type": "guideline",
                },
            ),
            Document(
                page_content="""2型糖尿病诊疗规范

诊断标准：
- 空腹血糖 ≥ 126 mg/dL (7.0 mmol/L)
- HbA1c ≥ 6.5%
- OGTT 2小时血糖 ≥ 200 mg/dL

治疗方案：
生活方式干预：医学营养治疗、规律运动、体重管理

药物治疗：
- 一线用药：二甲双胍（起始500mg，最大2550mg/天）
- 二线选择：SGLT2抑制剂、GLP-1受体激动剂、DPP-4抑制剂

血糖控制目标：
- HbA1c < 7%（大多数成人）""",
                metadata={
                    "source": "ADA糖尿病标准",
                    "title": "糖尿病管理指南",
                    "type": "guideline",
                },
            ),
            Document(
                page_content="""急性心肌梗死诊疗指南

临床表现：
- 胸痛或压迫感持续超过20分钟
- 疼痛可放射至左臂、下颌或背部
- 伴随症状：出汗、恶心、呼吸困难

诊断评估：
心电图：连续两个导联ST段抬高 ≥ 1mm（STEMI）
心肌标志物：肌钙蛋白I或T升高超过99百分位数

治疗方案：
STEMI：首次医疗接触90分钟内行直接PCI
NSTEMI：抗血小板治疗、抗凝、早期侵入性策略

二级预防用药：
- 阿司匹林 81mg 每日长期服用
- β受体阻滞剂
- ACE抑制剂或ARB
- 他汀类药物""",
                metadata={
                    "source": "ACC/AHA指南",
                    "title": "心肌梗死管理",
                    "type": "guideline",
                },
            ),
        ]

        self.add_documents(sample_docs)
        print(f"Loaded {len(sample_docs)} sample documents")


def create_rag_system(
    provider: str = "openai",
    model: str = None,
    api_key: str = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> MedicalRAGSystem:
    """
    Factory function to create RAG system.

    Args:
        provider: API provider (openai, zhipu, deepseek, moonshot)
        model: Model name
        api_key: API key
        embedding_model: Local embedding model (NO LLM)

    Returns:
        Initialized MedicalRAGSystem
    """
    config = MedicalRAGConfig(
        llm_provider=provider,
        llm_model=model,
        llm_api_key=api_key,
        embedding_model=embedding_model,
    )

    system = MedicalRAGSystem(config)
    system.initialize()

    return system


# ============================================================
# Convenience functions for different providers
# ============================================================


def create_openai_rag(
    model: str = "gpt-4o-mini", api_key: str = None
) -> MedicalRAGSystem:
    """Create RAG system with OpenAI"""
    return create_rag_system("openai", model, api_key)


def create_zhipu_rag(
    model: str = "glm-4-flash", api_key: str = None
) -> MedicalRAGSystem:
    """Create RAG system with Zhipu AI"""
    return create_rag_system("zhipu", model, api_key)


def create_deepseek_rag(
    model: str = "deepseek-chat", api_key: str = None
) -> MedicalRAGSystem:
    """Create RAG system with DeepSeek"""
    return create_rag_system("deepseek", model, api_key)


def create_rag_pipeline(
    provider: str = "openai",
    model: str = None,
    api_key: str = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> MedicalRAGSystem:
    """
    Factory function to create RAG pipeline.

    Alias for create_rag_system for backward compatibility.
    """
    return create_rag_system(provider, model, api_key, embedding_model)


if __name__ == "__main__":
    print("=" * 60)
    print("Medical RAG System - API Key Version")
    print("LLM ONLY used in Generator Module (via API)")
    print("=" * 60)

    # Check for API keys
    print("\nSupported API Providers:")
    print("-" * 40)

    providers = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Zhipu AI": os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY"),
        "DeepSeek": os.getenv("DEEPSEEK_API_KEY"),
        "Moonshot": os.getenv("MOONSHOT_API_KEY"),
    }

    for provider, key in providers.items():
        status = "✓ API Key Set" if key else "✗ API Key Not Set"
        print(f"  {provider}: {status}")

    print("\n" + "=" * 60)
    print("Architecture Summary:")
    print("=" * 60)
    print("┌─────────────────────────────────────────────┐")
    print("│ Module          │ Method          │ LLM?   │")
    print("├─────────────────────────────────────────────┤")
    print("│ Embedding       │ sentence-       │   NO   │")
    print("│                 │ transformers    │        │")
    print("├─────────────────────────────────────────────┤")
    print("│ Vector Store    │ FAISS           │   NO   │")
    print("├─────────────────────────────────────────────┤")
    print("│ Retrieval       │ Dense Search    │   NO   │")
    print("├─────────────────────────────────────────────┤")
    print("│ Generator       │ API LLM         │  YES   │")
    print("│                 │ (OpenAI/Zhipu)  │        │")
    print("├─────────────────────────────────────────────┤")
    print("│ Evaluation      │ Rule-based      │   NO   │")
    print("└─────────────────────────────────────────────┘")

    print("\nUsage Example:")
    print("-" * 40)
    print(
        """
from app.rag.api_medical_rag import create_openai_rag

# Set API key first
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Create and use system
system = create_openai_rag()
result = system.query("高血压的一线治疗方案是什么？")
print(result['answer'])
"""
    )
