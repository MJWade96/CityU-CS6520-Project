"""
Process MedRAG/textbooks dataset

Converts downloaded JSONL files into a unified JSON format for the RAG system.
"""

import os
import json
from pathlib import Path


def process_textbooks_data(
    input_dir: str = "./data/corpus/textbooks/chunk",
    output_dir: str = "./data/corpus/textbooks",
) -> None:
    """
    Process downloaded textbooks JSONL files
    
    Args:
        input_dir: Directory containing downloaded JSONL files
        output_dir: Directory to save processed data
    """
    print("=" * 60)
    print("Processing MedRAG/textbooks Dataset")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSONL files
    jsonl_files = list(Path(input_dir).glob("*.jsonl"))
    print(f"\nFound {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"  - {f.name}")
    
    # Process all files
    all_chunks = []
    total_size = 0
    
    for jsonl_file in jsonl_files:
        print(f"\nProcessing {jsonl_file.name}...")
        
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                file_chunks = 0
                for line in f:
                    if line.strip():
                        chunk_data = json.loads(line)
                        
                        # Convert to standard format
                        chunk = {
                            "id": chunk_data.get("id", ""),
                            "title": chunk_data.get("title", ""),
                            "content": chunk_data.get("content", ""),
                            "contents": chunk_data.get("contents", ""),  # For BM25
                            "source": "medrag_textbooks",
                            "textbook": jsonl_file.stem,  # e.g., "Anatomy_Gray"
                        }
                        
                        all_chunks.append(chunk)
                        file_chunks += 1
                
                print(f"  ✓ {file_chunks:,} chunks")
                total_size += os.path.getsize(jsonl_file)
        except Exception as e:
            print(f"  ✗ Error processing {jsonl_file.name}: {e}")
    
    print(f"\n✓ Total chunks: {len(all_chunks):,}")
    print(f"✓ Total size: {total_size / (1024*1024):.1f} MB")
    
    # Save combined JSON
    output_file = os.path.join(output_dir, "textbooks_combined.json")
    print(f"\nSaving to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved combined dataset")
    
    # Calculate statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    # Chunks per textbook
    textbook_counts = {}
    for chunk in all_chunks:
        textbook = chunk["textbook"]
        textbook_counts[textbook] = textbook_counts.get(textbook, 0) + 1
    
    print(f"\nChunks per textbook:")
    for textbook in sorted(textbook_counts.keys()):
        count = textbook_counts[textbook]
        print(f"  {textbook}: {count:,} chunks")
    
    # Length statistics
    content_lengths = [len(chunk["content"]) for chunk in all_chunks]
    avg_length = sum(content_lengths) // len(content_lengths)
    min_length = min(content_lengths)
    max_length = max(content_lengths)
    
    print(f"\nContent length statistics:")
    print(f"  Average: {avg_length:,} characters")
    print(f"  Min: {min_length:,} characters")
    print(f"  Max: {max_length:,} characters")
    
    # Total characters
    total_chars = sum(content_lengths)
    print(f"\nTotal characters: {total_chars:,}")
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {output_file}")
    print(f"  - Size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")


def main():
    """Main function"""
    process_textbooks_data()


if __name__ == "__main__":
    main()
