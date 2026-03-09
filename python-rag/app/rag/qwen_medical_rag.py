"""
Medical RAG System - Qwen2.5-7B Version
LLM only used in Generator Module

Architecture:
- Embedding: sentence-transformers (local, no LLM)
- Vector Store: FAISS (local)
- Retrieval: Dense retrieval (no LLM)
- Generator: Qwen2.5-7B (via Ollama or vLLM)
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

# LLM - Only in Generator
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama


@dataclass
class MedicalRAGConfig:
    """Configuration for Medical RAG System - Qwen2.5-7B Version"""
    
    # LLM Settings (Only used in Generator)
    llm_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_base_url: str = "http://localhost:11434"  # Ollama default
    
    # Embedding Settings (Local, no LLM)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # or "cuda"
    
    # Retrieval Settings (No LLM)
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    
    # Vector Store
    vector_store_path: str = "./data/vector_store"
    
    # Evaluation Settings (No LLM)
    use_llm_evaluation: bool = False  # Disable LLM-based evaluation


class LocalEmbeddingModel:
    """
    Local Embedding Model - No LLM Required
    
    Uses sentence-transformers for local embedding generation.
    """
    
    # Recommended models for medical text
    RECOMMENDED_MODELS = {
        'fast': 'sentence-transformers/all-MiniLM-L6-v2',      # 384 dim, fast
        'balanced': 'sentence-transformers/all-mpnet-base-v2', # 768 dim, balanced
        'medical': 'emilyalsentzer/Bio_ClinicalBERT',          # 768 dim, medical
        'multilingual': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # 384 dim, multi-language
    }
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize local embedding model.
        
        Args:
            model_name: HuggingFace model name
            device: "cpu" or "cuda"
        """
        self.model_name = model_name
        self.device = device
        
        # Initialize HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Get embedding dimension
        self._dimension = self._get_dimension()
    
    def _get_dimension(self) -> int:
        """Get embedding dimension"""
        test_embedding = self.embeddings.embed_query("test")
        return len(test_embedding)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embeddings.embed_query(text)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension
    
    def get_langchain_embeddings(self):
        """Get LangChain-compatible embeddings object"""
        return self.embeddings


class QwenGenerator:
    """
    Qwen2.5-7B Generator Module
    
    The ONLY module that uses LLM.
    Supports Ollama or vLLM deployment.
    """
    
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
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        """
        Initialize Qwen2.5-7B generator.
        
        Args:
            model: Model name (default: qwen2.5:7b)
            base_url: Ollama server URL
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Ollama LLM
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens
        )
        
        print(f"Initialized Qwen2.5-7B generator at {base_url}")
    
    def generate(
        self,
        question: str,
        contexts: List[str],
        sources: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate answer using Qwen2.5-7B.
        
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
            context=formatted_context,
            question=question
        )
        
        # Generate response
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"生成回答时出错: {str(e)}"
    
    def _format_context(
        self,
        contexts: List[str],
        sources: Optional[List[Dict]] = None
    ) -> str:
        """Format contexts for prompt"""
        if not contexts:
            return "未找到相关医学文献。"
        
        formatted_parts = []
        for i, ctx in enumerate(contexts, 1):
            if sources and i <= len(sources):
                source_info = sources[i-1]
                formatted_parts.append(
                    f"[来源 {i}: {source_info.get('source', '未知')}]\n"
                    f"{ctx}"
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
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality using statistical methods.
        
        Metrics:
        - Keyword overlap: Query keywords in retrieved docs
        - Length ratio: Retrieved content length
        - Diversity: Unique content ratio
        """
        metrics = {}
        
        # Keyword overlap
        query_words = set(query.lower().split())
        overlap_scores = []
        for doc in retrieved_docs:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
            overlap_scores.append(overlap)
        
        metrics['keyword_overlap'] = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0
        
        # Length ratio
        total_length = sum(len(doc.page_content) for doc in retrieved_docs)
        metrics['avg_content_length'] = total_length / len(retrieved_docs) if retrieved_docs else 0
        
        # Diversity (unique content)
        all_content = " ".join(doc.page_content for doc in retrieved_docs)
        unique_words = len(set(all_content.lower().split()))
        total_words = len(all_content.split())
        metrics['content_diversity'] = unique_words / total_words if total_words > 0 else 0
        
        return metrics
    
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate answer quality using rule-based methods.
        
        Metrics:
        - Context coverage: Answer words from context
        - Answer length: Appropriate length
        - Question relevance: Question keywords in answer
        """
        metrics = {}
        
        # Context coverage
        answer_words = set(answer.lower().split())
        context_words = set(" ".join(contexts).lower().split())
        coverage = len(answer_words & context_words) / len(answer_words) if answer_words else 0
        metrics['context_coverage'] = coverage
        
        # Answer length score (normalized)
        answer_len = len(answer.split())
        if answer_len < 20:
            metrics['length_score'] = 0.3  # Too short
        elif answer_len > 500:
            metrics['length_score'] = 0.7  # Might be too long
        else:
            metrics['length_score'] = 1.0  # Good length
        
        # Question relevance
        question_words = set(question.lower().split())
        question_in_answer = len(question_words & answer_words) / len(question_words) if question_words else 0
        metrics['question_relevance'] = question_in_answer
        
        # Ground truth comparison (if available)
        if ground_truth:
            gt_words = set(ground_truth.lower().split())
            gt_overlap = len(gt_words & answer_words) / len(gt_words) if gt_words else 0
            metrics['ground_truth_overlap'] = gt_overlap
        
        return metrics
    
    def check_safety(self, answer: str) -> Dict[str, Any]:
        """
        Check answer safety using rules.
        
        Checks:
        - Dangerous patterns
        - Required disclaimers
        - Appropriate certainty level
        """
        answer_lower = answer.lower()
        
        # Dangerous patterns
        dangerous_patterns = [
            "停止服药", "不要看医生", "一定可以治愈",
            "绝对安全", "没有任何副作用", "替代医生"
        ]
        found_dangerous = [p for p in dangerous_patterns if p in answer_lower]
        
        # Required disclaimers
        disclaimer_patterns = [
            "咨询", "医生", "医疗专业人员", "仅供参考"
        ]
        found_disclaimers = [p for p in disclaimer_patterns if p in answer_lower]
        
        # Calculate safety score
        safety_score = 1.0
        if found_dangerous:
            safety_score -= 0.3 * len(found_dangerous)
        if not found_disclaimers:
            safety_score -= 0.2
        
        safety_score = max(0.0, min(1.0, safety_score))
        
        return {
            'safety_score': safety_score,
            'dangerous_patterns_found': found_dangerous,
            'disclaimers_found': found_disclaimers,
            'is_safe': safety_score >= 0.7
        }
    
    def comprehensive_evaluation(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        retrieved_docs: List[Document],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation without LLM.
        """
        retrieval_metrics = self.evaluate_retrieval(question, retrieved_docs, ground_truth)
        answer_metrics = self.evaluate_answer(question, answer, contexts, ground_truth)
        safety_metrics = self.check_safety(answer)
        
        # Calculate overall score
        scores = [
            retrieval_metrics.get('keyword_overlap', 0),
            answer_metrics.get('context_coverage', 0),
            answer_metrics.get('question_relevance', 0),
            safety_metrics['safety_score']
        ]
        overall_score = sum(scores) / len(scores)
        
        return {
            'overall_score': overall_score,
            'retrieval': retrieval_metrics,
            'answer': answer_metrics,
            'safety': safety_metrics
        }


class MedicalRAGSystem:
    """
    Complete Medical RAG System
    
    Architecture:
    - Embedding: Local sentence-transformers (NO LLM)
    - Vector Store: FAISS (NO LLM)
    - Retrieval: Dense retrieval (NO LLM)
    - Generator: Qwen2.5-7B (ONLY module with LLM)
    - Evaluation: Rule-based (NO LLM)
    """
    
    def __init__(self, config: Optional[MedicalRAGConfig] = None):
        """Initialize the RAG system"""
        self.config = config or MedicalRAGConfig()
        
        # Initialize embedding model (NO LLM)
        self.embedding_model = LocalEmbeddingModel(
            model_name=self.config.embedding_model,
            device=self.config.embedding_device
        )
        
        # Initialize vector store (NO LLM)
        self.vector_store = None
        
        # Initialize generator (ONLY module with LLM)
        self.generator = QwenGenerator(
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
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
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(self.documents)
        
        # Build vector store (NO LLM)
        self.vector_store = FAISS.from_documents(
            split_docs,
            self.embedding_model.get_langchain_embeddings()
        )
        
        print(f"Built index with {len(split_docs)} chunks")
    
    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant documents (NO LLM)"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        k = k or self.config.top_k
        return self.vector_store.similarity_search(query, k=k)
    
    def generate(
        self,
        question: str,
        documents: List[Document]
    ) -> str:
        """Generate answer using Qwen2.5-7B (ONLY LLM usage)"""
        contexts = [doc.page_content for doc in documents]
        sources = [doc.metadata for doc in documents]
        
        return self.generator.generate(question, contexts, sources)
    
    def query(
        self,
        question: str,
        evaluate: bool = True
    ) -> Dict[str, Any]:
        """
        Process a medical query.
        
        Flow:
        1. Retrieve documents (NO LLM)
        2. Generate answer with Qwen2.5-7B (LLM)
        3. Evaluate results (NO LLM)
        """
        if not self.is_initialized:
            raise ValueError("System not initialized. Call initialize() first.")
        
        # Step 1: Retrieve (NO LLM)
        documents = self.retrieve(question)
        
        # Step 2: Generate (LLM - Qwen2.5-7B)
        answer = self.generate(question, documents)
        
        # Step 3: Evaluate (NO LLM)
        if evaluate:
            contexts = [doc.page_content for doc in documents]
            eval_results = self.evaluator.comprehensive_evaluation(
                question=question,
                answer=answer,
                contexts=contexts,
                retrieved_docs=documents
            )
        else:
            eval_results = None
        
        # Format sources
        sources = [
            {
                'source': doc.metadata.get('source', 'Unknown'),
                'title': doc.metadata.get('title', 'Unknown'),
                'content': doc.page_content[:200] + '...'
            }
            for doc in documents
        ]
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'evaluation': eval_results
        }
    
    def initialize(self, sample_data: bool = True) -> None:
        """Initialize the system"""
        if sample_data:
            self._load_sample_knowledge()
        
        if self.documents:
            self.build_index()
        
        self.is_initialized = True
        print("Medical RAG System initialized successfully")
        print(f"  - Embedding: {self.config.embedding_model} (Local, NO LLM)")
        print(f"  - LLM: {self.config.llm_model} (Generator ONLY)")
        print(f"  - Evaluation: Rule-based (NO LLM)")
    
    def _load_sample_knowledge(self) -> None:
        """Load sample medical knowledge"""
        sample_docs = [
            Document(
                page_content='''高血压临床实践指南

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

目标血压：大多数成人 < 130/80 mmHg''',
                metadata={'source': 'AHA/ACC指南', 'title': '高血压管理指南', 'type': 'guideline'}
            ),
            Document(
                page_content='''2型糖尿病诊疗规范

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
- HbA1c < 7%（大多数成人）''',
                metadata={'source': 'ADA糖尿病标准', 'title': '糖尿病管理指南', 'type': 'guideline'}
            ),
            Document(
                page_content='''急性心肌梗死诊疗指南

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
- 他汀类药物''',
                metadata={'source': 'ACC/AHA指南', 'title': '心肌梗死管理', 'type': 'guideline'}
            ),
        ]
        
        self.add_documents(sample_docs)
        print(f"Loaded {len(sample_docs)} sample documents")


def create_rag_system(
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model: str = "qwen2.5:7b",
    llm_base_url: str = "http://localhost:11434"
) -> MedicalRAGSystem:
    """
    Factory function to create RAG system.
    
    Args:
        embedding_model: Local embedding model (NO LLM)
        llm_model: Qwen model name
        llm_base_url: Ollama server URL
        
    Returns:
        Initialized MedicalRAGSystem
    """
    config = MedicalRAGConfig(
        embedding_model=embedding_model,
        llm_model=llm_model,
        llm_base_url=llm_base_url
    )
    
    system = MedicalRAGSystem(config)
    system.initialize()
    
    return system


if __name__ == "__main__":
    print("=" * 60)
    print("Medical RAG System - Qwen2.5-7B Version")
    print("LLM ONLY used in Generator Module")
    print("=" * 60)
    
    # Check Ollama availability
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("\n✓ Ollama server is running")
            models = response.json().get('models', [])
            qwen_available = any('qwen' in m.get('name', '').lower() for m in models)
            if qwen_available:
                print("✓ Qwen2.5-7B model is available")
            else:
                print("⚠ Qwen2.5-7B not found. Run: ollama pull qwen2.5:7b")
        else:
            print("⚠ Ollama server not responding properly")
    except Exception as e:
        print(f"⚠ Cannot connect to Ollama: {e}")
        print("  Please start Ollama: ollama serve")
        print("  And pull Qwen: ollama pull qwen2.5:7b")
    
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
    print("│ Generator       │ Qwen2.5-7B      │  YES   │")
    print("├─────────────────────────────────────────────┤")
    print("│ Evaluation      │ Rule-based      │   NO   │")
    print("└─────────────────────────────────────────────┘")
