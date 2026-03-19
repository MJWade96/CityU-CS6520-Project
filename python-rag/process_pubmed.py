"""
Process PubMed Abstracts for RAG System

Converts downloaded PubMed abstracts into chunks compatible with the RAG system.
Each abstract is treated as a single chunk (as per project design).
"""

import os
import json
from pathlib import Path


def process_pubmed_abstracts(
    input_file: str = "./data/corpus/pubmed/pubmed_abstracts.json",
    output_file: str = "./data/corpus/pubmed/pubmed_chunks.json",
):
    """
    Process PubMed abstracts into RAG-compatible format
    
    Args:
        input_file: Path to downloaded PubMed JSON file
        output_file: Path to save processed chunks
    """
    print("=" * 60)
    print("Processing PubMed Abstracts")
    print("=" * 60)
    
    # Load downloaded abstracts
    print(f"\nLoading from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        abstracts = json.load(f)
    
    print(f"✓ Loaded {len(abstracts)} abstracts")
    
    # Convert to RAG-compatible format
    chunks = []
    
    for i, abstract in enumerate(abstracts):
        # Create chunk from abstract
        chunk = {
            "id": f"pubmed_{abstract['pmid']}_{i}",
            "title": abstract["title"],
            "content": abstract["abstract"],
            "contents": f"{abstract['title']}. {abstract['abstract']}",
            "source": "pubmed",
            "pmid": abstract["pmid"],
            "journal": abstract.get("journal", ""),
            "year": abstract.get("year", ""),
            "authors": abstract.get("authors", []),
        }
        
        chunks.append(chunk)
    
    print(f"✓ Created {len(chunks)} chunks")
    
    # Save processed chunks
    print(f"\nSaving to {output_file}...")
    
    # Create directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    file_size = os.path.getsize(output_file)
    print(f"✓ Saved ({file_size / (1024*1024):.1f} MB)")
    
    # Statistics
    print(f"\n{'=' * 60}")
    print("Statistics:")
    
    # Content lengths
    content_lengths = [len(chunk["content"]) for chunk in chunks]
    avg_length = sum(content_lengths) // len(content_lengths)
    min_length = min(content_lengths)
    max_length = max(content_lengths)
    
    print(f"  Average chunk length: {avg_length:,} characters")
    print(f"  Min length: {min_length:,} characters")
    print(f"  Max length: {max_length:,} characters")
    
    # Years
    years = [chunk["year"] for chunk in chunks if chunk["year"]]
    if years:
        print(f"  Year range: {min(years)} - {max(years)}")
    
    # Journals
    journals = set(chunk["journal"] for chunk in chunks if chunk["journal"])
    print(f"  Unique journals: {len(journals)}")
    
    print(f"\n{'=' * 60}")
    print("SUCCESS!")
    print(f"{'=' * 60}")
    
    return len(chunks)


def main():
    """Main entry point"""
    import sys
    
    input_file = "./data/corpus/pubmed/pubmed_abstracts.json"
    output_file = "./data/corpus/pubmed/pubmed_chunks.json"
    
    if "--input" in sys.argv:
        idx = sys.argv.index("--input") + 1
        if idx < len(sys.argv):
            input_file = sys.argv[idx]
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output") + 1
        if idx < len(sys.argv):
            output_file = sys.argv[idx]
    
    # Process
    count = process_pubmed_abstracts(
        input_file=input_file,
        output_file=output_file
    )
    
    print(f"\nProcessed {count} PubMed abstracts")
    print(f"Output: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()
