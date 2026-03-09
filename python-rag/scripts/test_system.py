#!/usr/bin/env python3
"""
Test script for Medical RAG System
Run this to verify the installation and basic functionality
"""

import sys
import os

# Add the app directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from app.rag.config import RAGConfig, DEFAULT_CONFIG
        print("  ✓ config module")
    except ImportError as e:
        print(f"  ✗ config module: {e}")
        return False
    
    try:
        from app.rag.document_loader import MedicalDocumentLoader, load_sample_knowledge_base
        print("  ✓ document_loader module")
    except ImportError as e:
        print(f"  ✗ document_loader module: {e}")
        return False
    
    try:
        from app.rag.embeddings import MockEmbeddingModel, get_embedding_model
        print("  ✓ embeddings module")
    except ImportError as e:
        print(f"  ✗ embeddings module: {e}")
        return False
    
    try:
        from app.rag.vector_store import MedicalVectorStore
        print("  ✓ vector_store module")
    except ImportError as e:
        print(f"  ✗ vector_store module: {e}")
        return False
    
    try:
        from app.rag.pipeline import MedicalRAGPipeline, create_rag_pipeline
        print("  ✓ pipeline module")
    except ImportError as e:
        print(f"  ✗ pipeline module: {e}")
        return False
    
    return True


def test_document_loader():
    """Test document loading functionality"""
    print("\nTesting document loader...")
    
    from app.rag.document_loader import MedicalDocumentLoader, load_sample_knowledge_base
    
    # Test loader
    loader = MedicalDocumentLoader(chunk_size=512, chunk_overlap=50)
    print("  ✓ Document loader created")
    
    # Test knowledge base
    kb = load_sample_knowledge_base()
    stats = kb.get_stats()
    print(f"  ✓ Knowledge base loaded: {stats['total_documents']} documents")
    
    return True


def test_embeddings():
    """Test embedding functionality"""
    print("\nTesting embeddings...")
    
    from app.rag.embeddings import MockEmbeddingModel
    
    # Test mock embeddings
    model = MockEmbeddingModel(dimension=384)
    embedding = model.embed_query("test query")
    
    print(f"  ✓ Mock embedding dimension: {len(embedding)}")
    
    # Test document embedding
    docs = model.embed_documents(["doc1", "doc2"])
    print(f"  ✓ Embedded {len(docs)} documents")
    
    return True


def test_vector_store():
    """Test vector store functionality"""
    print("\nTesting vector store...")
    
    from langchain.docstore.document import Document
    from app.rag.embeddings import get_langchain_embeddings
    from app.rag.vector_store import MedicalVectorStore
    
    # Create mock embeddings
    embeddings = get_langchain_embeddings(model_type="mock")
    
    # Create vector store
    store = MedicalVectorStore(embeddings)
    
    # Add test documents
    docs = [
        Document(page_content="Hypertension is high blood pressure.", metadata={"category": "cardiovascular"}),
        Document(page_content="Diabetes affects blood sugar.", metadata={"category": "endocrine"}),
    ]
    
    store.add_documents(docs)
    print(f"  ✓ Added {len(docs)} documents to vector store")
    
    # Test search
    results = store.similarity_search("blood pressure", k=2)
    print(f"  ✓ Search returned {len(results)} results")
    
    return True


def test_pipeline():
    """Test the complete RAG pipeline"""
    print("\nTesting RAG pipeline...")
    
    from app.rag.pipeline import create_rag_pipeline
    
    # Create pipeline
    pipeline = create_rag_pipeline()
    print("  ✓ Pipeline created")
    
    # Get stats
    stats = pipeline.get_stats()
    print(f"  ✓ Pipeline stats: {stats}")
    
    # Test query
    result = pipeline.query("What is hypertension?")
    print(f"  ✓ Query processed in {result.total_time:.2f}s")
    print(f"  ✓ Confidence: {result.confidence:.2f}")
    print(f"  ✓ Sources: {len(result.sources)}")
    
    return True


def main():
    """Run all tests"""
    print("=" * 50)
    print("  Medical RAG System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Document Loader", test_document_loader),
        ("Embeddings", test_embeddings),
        ("Vector Store", test_vector_store),
        ("Pipeline", test_pipeline),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
