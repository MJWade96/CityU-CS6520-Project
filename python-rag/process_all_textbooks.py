"""
Complete pipeline to process MedRAG/textbooks dataset

This script:
1. Reads all JSONL files from textbooks/chunk/
2. Combines them into a single JSON file
3. Copies to statpearls directory for compatibility
"""

import os
import json
from pathlib import Path


def main():
    print("=" * 60)
    print("Processing MedRAG/textbooks Dataset")
    print("=" * 60)

    # Use absolute paths based on script location
    script_dir = Path(__file__).parent
    input_dir = script_dir / "data" / "corpus" / "textbooks" / "chunk"
    output_dir = script_dir / "data" / "corpus" / "textbooks"
    statpearls_dir = script_dir / "data" / "corpus" / "statpearls"

    print(f"Script directory: {script_dir}")
    print(f"Input directory: {input_dir}")

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(statpearls_dir, exist_ok=True)

    # Find all JSONL files
    jsonl_files = sorted(Path(input_dir).glob("*.jsonl"))
    print(f"\nFound {len(jsonl_files)} JSONL files")

    if not jsonl_files:
        print("Error: No JSONL files found in", input_dir)
        print("Make sure to download the data first:")
        print(
            "  huggingface-cli download MedRAG/textbooks --repo-type=dataset --local-dir data/corpus/textbooks"
        )
        return

    # Process all files
    all_chunks = []
    total_chars = 0

    for jsonl_file in jsonl_files:
        print(f"Processing {jsonl_file.stem}...")

        with open(jsonl_file, "r", encoding="utf-8") as f:
            file_chunks = 0
            for line in f:
                if line.strip():
                    chunk_data = json.loads(line)

                    # Standard format
                    chunk = {
                        "id": chunk_data.get("id", ""),
                        "title": chunk_data.get("title", ""),
                        "content": chunk_data.get("content", ""),
                        "contents": chunk_data.get("contents", ""),
                        "source": "medrag_textbooks",
                        "textbook": jsonl_file.stem,
                    }

                    all_chunks.append(chunk)
                    file_chunks += 1
                    total_chars += len(chunk["content"])

            print(f"  ✓ {file_chunks:,} chunks")

    print(f"\n{'=' * 60}")
    print(f"Total chunks: {len(all_chunks):,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average length: {total_chars // len(all_chunks):,} chars")

    # Save combined JSON
    output_file = os.path.join(output_dir, "textbooks_combined.json")
    print(f"\nSaving to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    file_size = os.path.getsize(output_file)
    print(f"✓ Saved ({file_size / (1024*1024):.1f} MB)")

    # Copy to statpearls directory for compatibility
    statpearls_file = os.path.join(statpearls_dir, "statpearls_articles.json")
    print(f"\nCopying to {statpearls_file} for RAG compatibility...")

    with open(statpearls_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"✓ Done!")

    # Statistics by textbook
    print(f"\n{'=' * 60}")
    print("Chunks per textbook:")
    print("=" * 60)

    textbook_counts = {}
    for chunk in all_chunks:
        textbook = chunk["textbook"]
        textbook_counts[textbook] = textbook_counts.get(textbook, 0) + 1

    for textbook in sorted(textbook_counts.keys()):
        count = textbook_counts[textbook]
        print(f"  {textbook}: {count:,} chunks")

    print(f"\n{'=' * 60}")
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. {output_file}")
    print(f"  2. {statpearls_file}")
    print(f"\nReady to use with RAG system!")


if __name__ == "__main__":
    main()
