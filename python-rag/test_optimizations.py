"""
Test script for Phase 1 and Phase 2 optimization modules

Tests all implemented modules:
1. Hybrid Retriever
2. Query Rewrite
3. Reranker
4. Chunking
5. Metadata Enhancement
6. Prompt Templates
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.documents import Document


def test_hybrid_retriever():
    """Test hybrid retrieval module"""
    print("\n" + "=" * 60)
    print("Testing Hybrid Retriever")
    print("=" * 60)
    
    try:
        from app.rag.hybrid_retriever import HybridRetriever, AdaptiveRetriever
        from app.rag.embeddings import get_langchain_embeddings
        
        # Create test documents
        docs = [
            Document(page_content="Hypertension is high blood pressure. Symptoms include headache and dizziness."),
            Document(page_content="Diabetes affects blood sugar levels. Treatment includes insulin."),
            Document(page_content="Pneumonia is a lung infection. Symptoms include cough and fever."),
            Document(page_content="Heart disease can cause chest pain and shortness of breath."),
            Document(page_content="Cancer treatment includes chemotherapy and radiation therapy."),
        ]
        
        # Initialize embeddings
        embeddings = get_langchain_embeddings(model_type="huggingface")
        
        # Create hybrid retriever
        retriever = HybridRetriever(
            embedding_model=embeddings,
            documents=docs,
            dense_weight=0.5
        )
        
        # Test search
        query = "What is high blood pressure?"
        results = retriever.search(query, k=3, use_hybrid=True)
        
        print(f"✓ Query: {query}")
        print(f"✓ Found {len(results)} results")
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. Score: {score:.4f} | {doc.page_content[:50]}...")
        
        # Test adaptive retriever
        adaptive = AdaptiveRetriever(retriever)
        should_retrieve = adaptive.should_retrieve("What are symptoms of hypertension?")
        print(f"✓ Adaptive retrieval test: {should_retrieve}")
        
        print("✓ Hybrid Retriever: PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Hybrid Retriever: FAILED - {e}\n")
        return False


def test_query_rewrite():
    """Test query rewrite module"""
    print("\n" + "=" * 60)
    print("Testing Query Rewrite")
    print("=" * 60)
    
    try:
        from app.rag.query_rewrite import MedicalDictionaryRewriter, QueryRewritePipeline
        
        # Test dictionary rewriter
        dict_rewriter = MedicalDictionaryRewriter()
        
        test_queries = [
            "What is MI?",
            "Treatment for high blood pressure",
        ]
        
        print("Dictionary-based rewriting:")
        for query in test_queries:
            rewritten = dict_rewriter.rewrite(query)
            print(f"  Original:  {query}")
            print(f"  Rewritten: {rewritten}")
        
        # Test pipeline
        pipeline = QueryRewritePipeline(
            use_dict=True,
            use_llm=False,  # Skip LLM for faster testing
            use_expansion=False,
        )
        
        primary, all_queries = pipeline.rewrite("What is MI?", mode='single')
        print(f"\nPipeline test:")
        print(f"  Primary: {primary}")
        
        print("✓ Query Rewrite: PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Query Rewrite: FAILED - {e}\n")
        return False


def test_reranker():
    """Test reranker module"""
    print("\n" + "=" * 60)
    print("Testing Reranker")
    print("=" * 60)
    
    try:
        from app.rag.reranker import MMReranker, LostInTheMiddleReranker, RerankerPipeline
        
        # Create test documents
        docs = [
            (Document(page_content="Hypertension is high blood pressure."), 0.9),
            (Document(page_content="Diabetes affects blood sugar."), 0.85),
            (Document(page_content="High blood pressure can cause headaches."), 0.8),
            (Document(page_content="Pneumonia is a lung infection."), 0.75),
            (Document(page_content="Blood pressure medication includes ACE inhibitors."), 0.7),
        ]
        
        query = "What is hypertension?"
        
        # Test MMR
        mmr = MMReranker(lambda_mult=0.7)
        selected = mmr.rerank(query, docs, top_k=3)
        print(f"✓ MMR Reranker: Selected {len(selected)} documents")
        
        # Test LostInTheMiddle
        litm = LostInTheMiddleReranker()
        ordered = litm.rerank(query, docs, top_k=5)
        print(f"✓ LostInTheMiddle: Ordered {len(ordered)} documents")
        
        # Test pipeline
        pipeline = RerankerPipeline(
            use_cross_encoder=False,  # Skip for faster testing
            use_mmr=True,
            use_lost_in_middle=False,
            top_k=3
        )
        
        stats = pipeline.get_stats()
        print(f"✓ Pipeline config: {stats}")
        
        print("✓ Reranker: PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Reranker: FAILED - {e}\n")
        return False


def test_chunking():
    """Test chunking module"""
    print("\n" + "=" * 60)
    print("Testing Chunking")
    print("=" * 60)
    
    try:
        from app.rag.chunking import SemanticChunker, SlidingWindowChunker, ParentChildChunker
        
        # Sample text
        sample_text = """
        Hypertension is a common medical condition characterized by elevated blood pressure in the arteries. 
        It is often referred to as high blood pressure.
        
        Hypertension is defined as having a systolic blood pressure of 130 mmHg or higher, or a diastolic blood pressure of 80 mmHg or higher.
        
        There are two types of hypertension:
        1. Primary (essential) hypertension: This is the most common type.
        2. Secondary hypertension: This type is caused by an underlying condition.
        
        Treatment options include lifestyle modifications and medications.
        """
        
        # Test semantic chunking
        semantic_chunker = SemanticChunker(chunk_size=200, chunk_overlap=30)
        chunks = semantic_chunker.chunk_text(sample_text, metadata={"source": "test"})
        print(f"✓ Semantic Chunking: Created {len(chunks)} chunks")
        
        # Test sliding window
        sliding_chunker = SlidingWindowChunker(chunk_size=300, chunk_overlap=50)
        chunks = sliding_chunker.chunk_text(sample_text)
        print(f"✓ Sliding Window: Created {len(chunks)} chunks")
        
        # Test parent-child chunking
        pc_chunker = ParentChildChunker(
            parent_chunk_size=400,
            child_chunk_size=150,
            child_overlap=30
        )
        parent_chunks, child_chunks = pc_chunker.chunk_text(sample_text)
        print(f"✓ Parent-Child: {len(parent_chunks)} parents, {len(child_chunks)} children")
        
        print("✓ Chunking: PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Chunking: FAILED - {e}\n")
        return False


def test_metadata_enhancement():
    """Test metadata enhancement module"""
    print("\n" + "=" * 60)
    print("Testing Metadata Enhancement")
    print("=" * 60)
    
    try:
        from app.rag.metadata_enhancement import RuleBasedMetadataGenerator
        
        sample_text = """
        Hypertension is a common medical condition characterized by elevated blood pressure in the arteries. 
        It is often referred to as high blood pressure. Hypertension is defined as having a systolic blood 
        pressure of 130 mmHg or higher, or a diastolic blood pressure of 80 mmHg or higher.
        
        Treatment options include lifestyle modifications and medications such as diuretics, ACE inhibitors, 
        ARBs, calcium channel blockers, and beta blockers.
        """
        
        # Test rule-based generator
        rule_generator = RuleBasedMetadataGenerator()
        
        metadata = rule_generator.generate_all(
            sample_text,
            include_summary=True,
            include_keywords=True,
            include_entities=True,
        )
        
        print(f"✓ Summary: {metadata.get('summary', 'N/A')[:100]}...")
        print(f"✓ Keywords: {metadata.get('keywords', [])}")
        print(f"✓ Entities: {metadata.get('entities', {})}")
        
        print("✓ Metadata Enhancement: PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Metadata Enhancement: FAILED - {e}\n")
        return False


def test_prompt_templates():
    """Test prompt template module"""
    print("\n" + "=" * 60)
    print("Testing Prompt Templates")
    print("=" * 60)
    
    try:
        from app.rag.prompt_template import create_rag_prompt_template
        
        # Test basic template
        basic_template = create_rag_prompt_template(template_type="medical")
        print(f"✓ Basic template created")
        
        # Test CoT template
        cot_template = create_rag_prompt_template(use_cot=True)
        print(f"✓ CoT template created")
        
        # Test structured template
        structured_template = create_rag_prompt_template(use_structured=True)
        print(f"✓ Structured template created")
        
        # Test few-shot template
        fewshot_template = create_rag_prompt_template(use_fewshot=True)
        print(f"✓ Few-shot template created")
        
        # Test formatting
        context = "Hypertension is high blood pressure."
        question = "What is hypertension?"
        options = "A. Low BP\nB. High BP\nC. Normal BP"
        
        formatted = basic_template.format(context=context, question=question, options=options)
        print(f"✓ Template formatting works")
        
        print("✓ Prompt Templates: PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Prompt Templates: FAILED - {e}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Phase 1 & Phase 2 Optimization Modules - Test Suite")
    print("=" * 60)
    
    results = {
        'Hybrid Retriever': test_hybrid_retriever(),
        'Query Rewrite': test_query_rewrite(),
        'Reranker': test_reranker(),
        'Chunking': test_chunking(),
        'Metadata Enhancement': test_metadata_enhancement(),
        'Prompt Templates': test_prompt_templates(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for module, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{module}: {status}")
    
    print(f"\nTotal: {passed}/{total} modules passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! Phase 1 and Phase 2 optimizations are ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
