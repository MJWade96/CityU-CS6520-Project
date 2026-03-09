"""
Test script for Medical RAG System
Run this script to test the system and generate results for the report
"""

import os
import sys
import json

# Add project path
sys.path.insert(0, '/home/z/my-project/python-rag')

def test_embedding_only():
    """Test embedding model without LLM"""
    print("=" * 60)
    print("Test 1: Embedding Model (No LLM Required)")
    print("=" * 60)
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        print("\nLoading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Test embedding
        test_texts = [
            "高血压的治疗方案",
            "糖尿病的诊断标准",
            "心肌梗死的临床表现"
        ]
        
        print("\nGenerating embeddings...")
        vectors = embeddings.embed_documents(test_texts)
        
        print(f"\n✓ Embedding model loaded successfully")
        print(f"  - Model: sentence-transformers/all-MiniLM-L6-v2")
        print(f"  - Dimension: {len(vectors[0])}")
        print(f"  - Test samples: {len(test_texts)}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_vector_store():
    """Test vector store without LLM"""
    print("\n" + "=" * 60)
    print("Test 2: Vector Store (No LLM Required)")
    print("=" * 60)
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        
        # Sample documents
        docs = [
            Document(page_content="高血压一线治疗药物包括噻嗪类利尿剂、钙通道阻滞剂、ACE抑制剂。", metadata={"source": "指南1"}),
            Document(page_content="糖尿病诊断标准：空腹血糖≥126mg/dL，HbA1c≥6.5%。", metadata={"source": "指南2"}),
            Document(page_content="心肌梗死典型表现为胸痛持续超过20分钟，可放射至左臂。", metadata={"source": "指南3"}),
        ]
        
        print("\nLoading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        print("Building vector store...")
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Test search
        query = "高血压怎么治疗"
        print(f"\nTest query: {query}")
        results = vector_store.similarity_search(query, k=2)
        
        print("\n✓ Vector store built successfully")
        print(f"  - Documents indexed: {len(docs)}")
        print(f"  - Search results for '{query}':")
        for i, doc in enumerate(results, 1):
            print(f"    {i}. {doc.page_content[:50]}...")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_rule_based_evaluation():
    """Test rule-based evaluation without LLM"""
    print("\n" + "=" * 60)
    print("Test 3: Rule-Based Evaluation (No LLM Required)")
    print("=" * 60)
    
    try:
        # Import from the module
        from app.rag.api_medical_rag import RuleBasedEvaluator
        from langchain_core.documents import Document
        
        evaluator = RuleBasedEvaluator()
        
        # Test data
        question = "高血压的一线治疗方案是什么？"
        answer = """根据临床指南，高血压的一线治疗方案包括：
1. 噻嗪类利尿剂（如氢氯噻嗪）
2. 钙通道阻滞剂（如氨氯地平）
3. ACE抑制剂（如赖诺普利）或ARB类药物

目标血压：大多数成人 < 130/80 mmHg

请咨询您的医疗专业人员获取个性化治疗建议。"""
        
        contexts = ["高血压一线治疗药物包括噻嗪类利尿剂、钙通道阻滞剂、ACE抑制剂。"]
        docs = [Document(page_content=c) for c in contexts]
        
        # Evaluate
        results = evaluator.comprehensive_evaluation(
            question=question,
            answer=answer,
            contexts=contexts,
            retrieved_docs=docs
        )
        
        print("\n✓ Evaluation completed successfully")
        print(f"\nEvaluation Results:")
        print(f"  - Overall Score: {results['overall_score']:.2f}")
        print(f"  - Keyword Overlap: {results['retrieval']['keyword_overlap']:.2f}")
        print(f"  - Context Coverage: {results['answer']['context_coverage']:.2f}")
        print(f"  - Question Relevance: {results['answer']['question_relevance']:.2f}")
        print(f"  - Safety Score: {results['safety']['safety_score']:.2f}")
        print(f"  - Is Safe: {results['safety']['is_safe']}")
        
        return True, results
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_full_system_mock():
    """Test full system with mock LLM response"""
    print("\n" + "=" * 60)
    print("Test 4: Full System Test (Mock LLM Response)")
    print("=" * 60)
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        from app.rag.api_medical_rag import RuleBasedEvaluator
        
        # Sample knowledge base
        docs = [
            Document(
                page_content='''高血压临床实践指南

血压分类：
- 正常血压：收缩压 < 120 mmHg 且 舒张压 < 80 mmHg
- 1级高血压：收缩压 130-139 mmHg 或 舒张压 80-89 mmHg
- 2级高血压：收缩压 ≥ 140 mmHg 或 舒张压 ≥ 90 mmHg

治疗方案：
一线药物包括：噻嗪类利尿剂、钙通道阻滞剂、ACE抑制剂或ARB类药物
目标血压：大多数成人 < 130/80 mmHg''',
                metadata={'source': 'AHA/ACC指南', 'title': '高血压管理指南'}
            ),
            Document(
                page_content='''2型糖尿病诊疗规范

诊断标准：
- 空腹血糖 ≥ 126 mg/dL (7.0 mmol/L)
- HbA1c ≥ 6.5%

治疗方案：
- 一线用药：二甲双胍
- 二线选择：SGLT2抑制剂、GLP-1受体激动剂''',
                metadata={'source': 'ADA糖尿病标准', 'title': '糖尿病管理指南'}
            ),
        ]
        
        print("\n1. Building embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        print("2. Building vector store...")
        vector_store = FAISS.from_documents(docs, embeddings)
        
        print("3. Testing retrieval...")
        query = "高血压的一线治疗方案是什么？"
        retrieved = vector_store.similarity_search(query, k=2)
        
        print(f"\n   Query: {query}")
        print(f"   Retrieved {len(retrieved)} documents")
        
        # Mock LLM response (simulating what the API would return)
        mock_answer = """根据临床指南，高血压的一线治疗方案包括：

1. 噻嗪类利尿剂（如氢氯噻嗪）
2. 钙通道阻滞剂（如氨氯地平）
3. ACE抑制剂（如赖诺普利）或ARB类药物（如氯沙坦）

目标血压：大多数成人 < 130/80 mmHg

请咨询您的医疗专业人员获取个性化治疗建议。"""
        
        print("\n4. Mock LLM Response:")
        print("-" * 40)
        print(mock_answer)
        print("-" * 40)
        
        # Evaluate
        print("\n5. Evaluating response...")
        evaluator = RuleBasedEvaluator()
        eval_results = evaluator.comprehensive_evaluation(
            question=query,
            answer=mock_answer,
            contexts=[doc.page_content for doc in retrieved],
            retrieved_docs=retrieved
        )
        
        print("\n✓ Full system test completed")
        print(f"\nFinal Evaluation:")
        print(f"  - Overall Score: {eval_results['overall_score']:.2f}")
        print(f"  - Safety Score: {eval_results['safety']['safety_score']:.2f}")
        
        return True, {
            'query': query,
            'answer': mock_answer,
            'sources': [doc.metadata for doc in retrieved],
            'evaluation': eval_results
        }
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def run_all_tests():
    """Run all tests and collect results"""
    print("\n" + "=" * 60)
    print("Medical RAG System - Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Embedding
    results['embedding'] = test_embedding_only()
    
    # Test 2: Vector Store
    results['vector_store'] = test_vector_store()
    
    # Test 3: Evaluation
    success, eval_results = test_rule_based_evaluation()
    results['evaluation'] = success
    results['eval_details'] = eval_results
    
    # Test 4: Full system
    success, full_results = test_full_system_mock()
    results['full_system'] = success
    results['full_details'] = full_results
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test, passed in results.items():
        if test.endswith('_details'):
            continue
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test}: {status}")
    
    # Save results
    output = {
        'test_results': {k: v for k, v in results.items() if not k.endswith('_details')},
        'evaluation_details': results.get('eval_details'),
        'full_system_details': results.get('full_details')
    }
    
    with open('/home/z/my-project/download/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\nResults saved to: /home/z/my-project/download/test_results.json")
    
    return results


if __name__ == "__main__":
    run_all_tests()
