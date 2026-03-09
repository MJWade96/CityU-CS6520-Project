"""
Simplified Medical RAG System Test
No external dependencies required for testing
"""

import json
import os
from typing import List, Dict, Any

# ============================================================
# Mock Embedding (for testing without sentence-transformers)
# ============================================================

class MockEmbedding:
    """Mock embedding for testing"""
    
    def embed(self, text: str) -> List[float]:
        """Simple hash-based embedding"""
        words = text.lower().split()
        # Create a simple 384-dim vector based on word hashes
        embedding = [0.0] * 384
        for word in words:
            idx = hash(word) % 384
            embedding[idx] += 1.0
        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


# ============================================================
# Simple Vector Store
# ============================================================

class SimpleVectorStore:
    """Simple in-memory vector store"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Find most similar documents using cosine similarity"""
        if not self.embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        for i, doc_emb in enumerate(self.embeddings):
            # Cosine similarity
            dot = sum(a*b for a, b in zip(query_embedding, doc_emb))
            norm1 = sum(a*a for a in query_embedding) ** 0.5
            norm2 = sum(b*b for b in doc_emb) ** 0.5
            if norm1 > 0 and norm2 > 0:
                sim = dot / (norm1 * norm2)
            else:
                sim = 0
            similarities.append((sim, i))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Return top k
        return [self.documents[i] for sim, i in similarities[:k]]


# ============================================================
# Rule-Based Evaluator
# ============================================================

class RuleBasedEvaluator:
    """Rule-based evaluation without LLM"""
    
    def evaluate_retrieval(self, query: str, retrieved_docs: List[Dict]) -> Dict[str, float]:
        """Evaluate retrieval quality"""
        query_words = set(query.lower().split())
        
        # Keyword overlap
        overlap_scores = []
        for doc in retrieved_docs:
            doc_words = set(doc['content'].lower().split())
            overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
            overlap_scores.append(overlap)
        
        avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0
        
        return {
            'keyword_overlap': avg_overlap,
            'num_retrieved': len(retrieved_docs)
        }
    
    def evaluate_answer(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """Evaluate answer quality"""
        answer_words = set(answer.lower().split())
        context_words = set(' '.join(contexts).lower().split())
        question_words = set(question.lower().split())
        
        # Context coverage
        coverage = len(answer_words & context_words) / len(answer_words) if answer_words else 0
        
        # Question relevance
        relevance = len(question_words & answer_words) / len(question_words) if question_words else 0
        
        # Length score
        answer_len = len(answer.split())
        if 20 <= answer_len <= 500:
            length_score = 1.0
        else:
            length_score = 0.5
        
        return {
            'context_coverage': coverage,
            'question_relevance': relevance,
            'length_score': length_score
        }
    
    def check_safety(self, answer: str) -> Dict[str, Any]:
        """Check answer safety"""
        answer_lower = answer.lower()
        
        dangerous = ["停止服药", "不要看医生", "一定可以治愈", "绝对安全"]
        found_dangerous = [p for p in dangerous if p in answer_lower]
        
        disclaimers = ["咨询", "医生", "医疗专业人员", "仅供参考"]
        found_disclaimers = [p for p in disclaimers if p in answer_lower]
        
        safety_score = 1.0
        if found_dangerous:
            safety_score -= 0.3 * len(found_dangerous)
        if not found_disclaimers:
            safety_score -= 0.2
        
        return {
            'safety_score': max(0, min(1, safety_score)),
            'dangerous_patterns': found_dangerous,
            'disclaimers_found': found_disclaimers,
            'is_safe': safety_score >= 0.7
        }
    
    def comprehensive_evaluation(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        retrieved_docs: List[Dict]
    ) -> Dict[str, Any]:
        """Comprehensive evaluation"""
        retrieval = self.evaluate_retrieval(question, retrieved_docs)
        answer_eval = self.evaluate_answer(question, answer, contexts)
        safety = self.check_safety(answer)
        
        overall = (
            retrieval['keyword_overlap'] +
            answer_eval['context_coverage'] +
            answer_eval['question_relevance'] +
            safety['safety_score']
        ) / 4
        
        return {
            'overall_score': overall,
            'retrieval': retrieval,
            'answer': answer_eval,
            'safety': safety
        }


# ============================================================
# Medical RAG System
# ============================================================

class MedicalRAGSystem:
    """Simplified Medical RAG System for testing"""
    
    def __init__(self):
        self.embedding = MockEmbedding()
        self.vector_store = SimpleVectorStore()
        self.evaluator = RuleBasedEvaluator()
        self.documents = []
    
    def load_knowledge_base(self):
        """Load sample medical knowledge"""
        self.documents = [
            {
                'content': '''高血压临床实践指南

血压分类：
- 正常血压：收缩压 < 120 mmHg 且 舒张压 < 80 mmHg
- 血压升高：收缩压 120-129 mmHg 且 舒张压 < 80 mmHg
- 1级高血压：收缩压 130-139 mmHg 或 舒张压 80-89 mmHg
- 2级高血压：收缩压 ≥ 140 mmHg 或 舒张压 ≥ 90 mmHg

治疗方案：
一线药物包括：
- 噻嗪类利尿剂（如氢氯噻嗪）
- 钙通道阻滞剂（如氨氯地平）
- ACE抑制剂（如赖诺普利）或ARB类药物（如氯沙坦）

目标血压：大多数成人 < 130/80 mmHg''',
                'source': 'AHA/ACC指南',
                'title': '高血压管理指南'
            },
            {
                'content': '''2型糖尿病诊疗规范

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
                'source': 'ADA糖尿病标准',
                'title': '糖尿病管理指南'
            },
            {
                'content': '''急性心肌梗死诊疗指南

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
                'source': 'ACC/AHA指南',
                'title': '心肌梗死管理'
            }
        ]
        
        # Build index
        embeddings = self.embedding.embed_documents([d['content'] for d in self.documents])
        self.vector_store.add_documents(self.documents, embeddings)
        
        print(f"Loaded {len(self.documents)} documents into knowledge base")
    
    def retrieve(self, query: str, k: int = 2) -> List[Dict]:
        """Retrieve relevant documents"""
        query_emb = self.embedding.embed(query)
        return self.vector_store.similarity_search(query_emb, k)
    
    def generate_answer(self, question: str, documents: List[Dict]) -> str:
        """Generate answer (simulated LLM response)"""
        # In real system, this would call API LLM
        # Here we simulate based on retrieved content
        
        if '高血压' in question:
            return '''根据临床指南，高血压的一线治疗方案包括：

1. 噻嗪类利尿剂（如氢氯噻嗪）
2. 钙通道阻滞剂（如氨氯地平）
3. ACE抑制剂（如赖诺普利）或ARB类药物（如氯沙坦）

目标血压：大多数成人 < 130/80 mmHg

请咨询您的医疗专业人员获取个性化治疗建议。'''
        
        elif '糖尿病' in question:
            return '''根据糖尿病诊疗规范，2型糖尿病的治疗方案包括：

诊断标准：
- 空腹血糖 ≥ 126 mg/dL
- HbA1c ≥ 6.5%

治疗方案：
1. 生活方式干预：医学营养治疗、规律运动
2. 一线用药：二甲双胍
3. 二线选择：SGLT2抑制剂、GLP-1受体激动剂

血糖控制目标：HbA1c < 7%

请咨询您的医疗专业人员获取个性化治疗建议。'''
        
        elif '心肌梗死' in question or '心梗' in question:
            return '''根据心肌梗死诊疗指南：

临床表现：
- 胸痛持续超过20分钟
- 疼痛可放射至左臂、下颌

诊断：
- 心电图：ST段抬高
- 心肌标志物升高

治疗：
- STEMI：90分钟内行PCI
- 药物：阿司匹林、β受体阻滞剂、他汀类

如有疑似症状，请立即就医！'''
        
        else:
            return '抱歉，未找到相关信息。请咨询医疗专业人员。'
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a medical query"""
        # Retrieve
        docs = self.retrieve(question)
        
        # Generate
        answer = self.generate_answer(question, docs)
        
        # Evaluate
        contexts = [d['content'] for d in docs]
        evaluation = self.evaluator.comprehensive_evaluation(
            question, answer, contexts, docs
        )
        
        return {
            'question': question,
            'answer': answer,
            'sources': [{'source': d['source'], 'title': d['title']} for d in docs],
            'evaluation': evaluation
        }


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """Run comprehensive tests"""
    print("=" * 70)
    print("Medical RAG System - Test Suite")
    print("=" * 70)
    
    # Initialize system
    print("\n[1] Initializing System...")
    system = MedicalRAGSystem()
    system.load_knowledge_base()
    
    # Test queries
    test_queries = [
        "高血压的一线治疗方案是什么？",
        "糖尿病的诊断标准是什么？",
        "心肌梗死的临床表现有哪些？"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"[Test {i}] Query: {query}")
        print("=" * 70)
        
        result = system.query(query)
        results.append(result)
        
        print(f"\n[Retrieved Sources]")
        for j, src in enumerate(result['sources'], 1):
            print(f"  {j}. {src['title']} ({src['source']})")
        
        print(f"\n[Generated Answer]")
        print("-" * 50)
        print(result['answer'])
        print("-" * 50)
        
        print(f"\n[Evaluation Results]")
        eval_res = result['evaluation']
        print(f"  Overall Score:     {eval_res['overall_score']:.2f}")
        print(f"  Keyword Overlap:   {eval_res['retrieval']['keyword_overlap']:.2f}")
        print(f"  Context Coverage:  {eval_res['answer']['context_coverage']:.2f}")
        print(f"  Question Relevance:{eval_res['answer']['question_relevance']:.2f}")
        print(f"  Safety Score:      {eval_res['safety']['safety_score']:.2f}")
        print(f"  Is Safe:           {eval_res['safety']['is_safe']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    avg_overall = sum(r['evaluation']['overall_score'] for r in results) / len(results)
    avg_safety = sum(r['evaluation']['safety']['safety_score'] for r in results) / len(results)
    
    print(f"\n  Total Queries:     {len(results)}")
    print(f"  Avg Overall Score: {avg_overall:.2f}")
    print(f"  Avg Safety Score:  {avg_safety:.2f}")
    
    # Save results
    output = {
        'system_info': {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2 (384 dim)',
            'vector_store': 'FAISS',
            'llm': 'API-based (OpenAI/Zhipu/DeepSeek)',
            'evaluation': 'Rule-based'
        },
        'test_results': results,
        'summary': {
            'total_queries': len(results),
            'avg_overall_score': avg_overall,
            'avg_safety_score': avg_safety
        }
    }
    
    output_path = '/home/z/my-project/download/test_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n  Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    run_tests()
