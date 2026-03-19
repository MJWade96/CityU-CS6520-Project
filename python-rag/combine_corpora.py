"""
Combine all corpora (Textbooks + PubMed) into a single dataset

Creates a unified corpus for the RAG system.
"""

import os
import json
from pathlib import Path


def combine_corpora(
    textbooks_file: str = "./data/corpus/textbooks/textbooks_combined.json",
    pubmed_file: str = "./data/corpus/pubmed/pubmed_chunks.json",
    output_file: str = "./data/corpus/combined_corpus.json",
):
    """
    Combine Textbooks and PubMed corpora
    
    Args:
        textbooks_file: Path to processed textbooks JSON
        pubmed_file: Path to processed PubMed JSON
        output_file: Path to save combined corpus
    """
    print("=" * 60)
    print("Combining Medical Corpora")
    print("=" * 60)
    
    all_chunks = []
    
    # Load Textbooks
    print(f"\nLoading Textbooks from {textbooks_file}...")
    if os.path.exists(textbooks_file):
        with open(textbooks_file, "r", encoding="utf-8") as f:
            textbooks = json.load(f)
        
        print(f"✓ Loaded {len(textbooks):,} textbook chunks")
        all_chunks.extend(textbooks)
    else:
        print("⚠ Textbooks file not found, skipping...")
    
    # Load PubMed
    print(f"\nLoading PubMed from {pubmed_file}...")
    if os.path.exists(pubmed_file):
        with open(pubmed_file, "r", encoding="utf-8") as f:
            pubmed = json.load(f)
        
        print(f"✓ Loaded {len(pubmed):,} PubMed abstracts")
        all_chunks.extend(pubmed)
    else:
        print("⚠ PubMed file not found, skipping...")
    
    if not all_chunks:
        print("\n❌ ERROR: No data loaded!")
        return 0
    
    # Statistics
    print(f"\n{'=' * 60}")
    print(f"Combined Corpus Statistics:")
    print(f"{'=' * 60}")
    print(f"Total chunks: {len(all_chunks):,}")
    
    # Count by source
    source_counts = {}
    for chunk in all_chunks:
        source = chunk["source"]
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\nBy source:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count:,} chunks")
    
    # Content lengths
    content_lengths = [len(chunk["content"]) for chunk in all_chunks]
    avg_length = sum(content_lengths) // len(content_lengths)
    
    print(f"\nContent length:")
    print(f"  Average: {avg_length:,} characters")
    print(f"  Min: {min(content_lengths):,} characters")
    print(f"  Max: {max(content_lengths):,} characters")
    
    # Total size
    total_chars = sum(content_lengths)
    print(f"\nTotal characters: {total_chars:,}")
    
    # Save combined corpus
    print(f"\n{'=' * 60}")
    print(f"Saving to {output_file}...")
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    file_size = os.path.getsize(output_file)
    print(f"✓ Saved ({file_size / (1024*1024):.1f} MB)")
    
    print(f"\n{'=' * 60}")
    print("SUCCESS!")
    print(f"{'=' * 60}")
    
    return len(all_chunks)


def main():
    """Main entry point"""
    import sys
    
    # Default paths
    textbooks_file = "./data/corpus/textbooks/textbooks_combined.json"
    pubmed_file = "./data/corpus/pubmed/pubmed_chunks.json"
    output_file = "./data/corpus/combined_corpus.json"
    
    # Parse arguments
    if "--textbooks" in sys.argv:
        idx = sys.argv.index("--textbooks") + 1
        if idx < len(sys.argv):
            textbooks_file = sys.argv[idx]
    
    if "--pubmed" in sys.argv:
        idx = sys.argv.index("--pubmed") + 1
        if idx < len(sys.argv):
            pubmed_file = sys.argv[idx]
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output") + 1
        if idx < len(sys.argv):
            output_file = sys.argv[idx]
    
    # Combine
    count = combine_corpora(
        textbooks_file=textbooks_file,
        pubmed_file=pubmed_file,
        output_file=output_file
    )
    
    print(f"\nCombined corpus ready: {count:,} chunks")
    print(f"Output: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()
