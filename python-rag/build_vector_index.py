"""
Build Vector Index for Medical RAG System

This script:
1. Loads the combined corpus (Textbooks + PubMed)
2. Creates BGE embeddings
3. Builds FAISS vector index
4. Saves the index for later use

Usage:
    python build_vector_index.py
"""

import os
import sys
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.documents import Document
from app.rag.embeddings import get_langchain_embeddings
from app.rag.vector_store import MedicalVectorStore


def load_corpus(corpus_file: str = "./data/corpus/combined_corpus.json"):
    """
    Load the combined corpus
    
    Args:
        corpus_file: Path to combined corpus JSON file
    
    Returns:
        List of Document objects
    """
    print(f"Loading corpus from {corpus_file}...")
    
    if not os.path.exists(corpus_file):
        print(f"ERROR: Corpus file not found: {corpus_file}")
        return []
    
    with open(corpus_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"✓ Loaded {len(chunks):,} chunks")
    
    # Convert to LangChain documents
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["content"],
            metadata={
                "id": chunk.get("id", ""),
                "title": chunk.get("title", ""),
                "source": chunk.get("source", ""),
                "textbook": chunk.get("textbook", ""),
                "pmid": chunk.get("pmid", ""),
                "journal": chunk.get("journal", ""),
                "year": chunk.get("year", ""),
            }
        )
        documents.append(doc)
    
    print(f"✓ Created {len(documents):,} Document objects")
    return documents


def create_embeddings():
    """
    Create embeddings using Sentence Transformer
    
    Returns:
        LangChain embeddings instance
    """
    print("\nInitializing embedding model...")
    print("Model: all-MiniLM-L6-v2 (fast and effective)")
    
    # Use a lightweight but effective model
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("✓ Embedding model initialized")
    return embeddings


def build_vector_index(
    documents,
    embeddings,
    output_dir: str = "./data/vector_store/faiss_index"
):
    """
    Build FAISS vector index
    
    Args:
        documents: List of Document objects
        embeddings: LangChain embeddings instance
        output_dir: Directory to save the index
    
    Returns:
        MedicalVectorStore instance
    """
    print(f"\nBuilding FAISS vector index...")
    print(f"Total documents: {len(documents):,}")
    
    # Create vector store
    vectorstore = MedicalVectorStore(
        embedding_model=embeddings,
        store_type="faiss",
        persist_directory=output_dir
    )
    
    # Add documents in batches
    batch_size = 1000
    start_time = time.time()
    
    print(f"\nAdding documents in batches of {batch_size}...")
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        end_idx = min(i + batch_size, len(documents))
        
        vectorstore.add_documents(batch)
        
        # Progress update
        elapsed = time.time() - start_time
        docs_per_sec = end_idx / elapsed if elapsed > 0 else 0
        print(f"  Processed {end_idx:,}/{len(documents):,} docs "
              f"({elapsed:.1f}s, {docs_per_sec:.1f} docs/s)")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Index built in {elapsed:.1f} seconds")
    
    # Save the index
    print(f"\nSaving index to {output_dir}...")
    vectorstore.save(output_dir)
    
    # Save metadata
    metadata = {
        "document_count": len(documents),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "store_type": "faiss",
        "sources": {
            "medrag_textbooks": sum(1 for d in documents if d.metadata.get("source") == "medrag_textbooks"),
            "pubmed": sum(1 for d in documents if d.metadata.get("source") == "pubmed"),
        },
        "build_time_seconds": elapsed,
    }
    
    metadata_file = os.path.join(output_dir, "build_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Index saved")
    print(f"✓ Metadata saved to {metadata_file}")
    
    return vectorstore


def test_retrieval(vectorstore, k: int = 5):
    """
    Test the retrieval system
    
    Args:
        vectorstore: MedicalVectorStore instance
        k: Number of results to retrieve
    """
    print(f"\n{'=' * 60}")
    print("Testing Retrieval System")
    print(f"{'=' * 60}")
    
    # Test queries
    test_queries = [
        "What is hypertension?",
        "How to diagnose diabetes?",
        "Treatment for pneumonia",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            title = doc.metadata.get("title", "")[:50]
            
            print(f"{i}. [{source}] {title}")
            print(f"   Score: {score:.4f}")
            print(f"   Content: {doc.page_content[:100]}...")
        
        print()


def main():
    """Main entry point"""
    print("=" * 60)
    print("Building Medical RAG Vector Index")
    print("=" * 60)
    
    # Configuration
    corpus_file = "./data/corpus/combined_corpus.json"
    output_dir = "./data/vector_store/faiss_index"
    test_queries = True
    
    # Parse command line arguments
    if "--corpus" in sys.argv:
        idx = sys.argv.index("--corpus") + 1
        if idx < len(sys.argv):
            corpus_file = sys.argv[idx]
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output") + 1
        if idx < len(sys.argv):
            output_dir = sys.argv[idx]
    
    if "--no-test" in sys.argv:
        test_queries = False
    
    # Load corpus
    documents = load_corpus(corpus_file)
    
    if not documents:
        print("\n❌ ERROR: No documents loaded. Exiting...")
        return
    
    # Create embeddings
    embeddings = create_embeddings()
    
    # Build index
    vectorstore = build_vector_index(documents, embeddings, output_dir)
    
    # Test retrieval
    if test_queries:
        test_retrieval(vectorstore, k=5)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Vector Index Build Complete!")
    print(f"{'=' * 60}")
    print(f"Documents indexed: {len(documents):,}")
    print(f"Index location: {os.path.abspath(output_dir)}")
    print(f"Embedding model: BAAI/bge-small-en-v1.5")
    print(f"Vector store type: FAISS")
    print(f"\nReady to use with RAG system!")


if __name__ == "__main__":
    main()
