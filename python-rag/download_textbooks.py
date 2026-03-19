"""
Download MedRAG/textbooks Dataset from Hugging Face

This dataset contains medical textbook content used in MedRAG.
Reference: https://huggingface.co/datasets/MedRAG/textbooks

Usage:
    python download_textbooks.py
"""

import os
import json
from pathlib import Path


def download_medrag_textbooks(
    output_dir: str = "./data/corpus/textbooks",
    max_samples: int = None,
) -> None:
    """
    Download MedRAG/textbooks dataset from Hugging Face
    
    Args:
        output_dir: Directory to save the processed data
        max_samples: Maximum number of samples to download (None for all)
    """
    print("=" * 60)
    print("Downloading MedRAG/textbooks Dataset")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed")
        print("Install with: pip install datasets")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Load dataset from Hugging Face
    print("\nLoading MedRAG/textbooks dataset...")
    try:
        dataset = load_dataset("MedRAG/textbooks")
        print(f"✓ Dataset loaded successfully!")
        print(f"  Available splits: {list(dataset.keys())}")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process the dataset
    all_chunks = []
    
    for split_name, split_data in dataset.items():
        print(f"\nProcessing {split_name} split...")
        
        for i, item in enumerate(split_data):
            if max_samples and i >= max_samples:
                break
                
            chunk = {
                "id": item.get("id", f"{split_name}_{i}"),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "contents": item.get("contents", ""),  # For BM25 retrieval
                "source": "medrag_textbooks",
                "split": split_name,
            }
            
            all_chunks.append(chunk)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1:,} chunks...")
    
    print(f"\n✓ Total chunks processed: {len(all_chunks):,}")
    
    # Save as JSON
    output_file = os.path.join(output_dir, "textbooks_chunks.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved to: {output_file}")
    print(f"  Absolute path: {os.path.abspath(output_file)}")
    
    # Also save each textbook as separate markdown file
    markdown_dir = os.path.join(output_dir, "markdown")
    os.makedirs(markdown_dir, exist_ok=True)
    
    print(f"\nSaving individual markdown files to: {markdown_dir}")
    
    # Group by textbook (extract from id)
    textbooks = {}
    for chunk in all_chunks:
        # Extract textbook name from id (e.g., "Anatomy_Gray_0" -> "Anatomy_Gray")
        chunk_id = chunk["id"]
        parts = chunk_id.rsplit("_", 1)
        if len(parts) == 2:
            textbook_name = parts[0]
        else:
            textbook_name = "unknown"
        
        if textbook_name not in textbooks:
            textbooks[textbook_name] = []
        textbooks[textbook_name].append(chunk)
    
    # Save each textbook
    saved_count = 0
    for textbook_name, chunks in textbooks.items():
        # Sort chunks by id
        chunks.sort(key=lambda x: x["id"])
        
        # Combine all content
        md_content = f"# {textbook_name.replace('_', ' ')}\n\n"
        md_content += f"**Source:** MedRAG Textbooks\n"
        md_content += f"**Total chunks:** {len(chunks)}\n\n"
        md_content += "---\n\n"
        
        for chunk in chunks:
            md_content += f"## {chunk['title']}\n\n"
            md_content += f"{chunk['content']}\n\n"
            md_content += "---\n\n"
        
        md_file = os.path.join(markdown_dir, f"{textbook_name}.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        saved_count += 1
        if saved_count % 10 == 0:
            print(f"  Saved {saved_count} textbooks...")
    
    print(f"✓ Saved {saved_count} textbook files")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total chunks: {len(all_chunks):,}")
    print(f"Number of textbooks: {len(textbooks)}")
    
    # Calculate average length
    total_chars = sum(len(chunk["content"]) for chunk in all_chunks)
    avg_chars = total_chars // len(all_chunks) if all_chunks else 0
    print(f"Average chunk length: {avg_chars} characters")
    print(f"Total characters: {total_chars:,}")
    
    # List textbooks
    print(f"\nTextbooks included:")
    for textbook_name in sorted(textbooks.keys()):
        print(f"  - {textbook_name.replace('_', ' ')} ({len(textbooks[textbook_name])} chunks)")


def main():
    """Main function"""
    import sys
    
    output_dir = "./data/corpus/textbooks"
    max_samples = None
    
    # Parse command line arguments
    if "--max-samples" in sys.argv:
        idx = sys.argv.index("--max-samples") + 1
        if idx < len(sys.argv):
            max_samples = int(sys.argv[idx])
            print(f"Will download only {max_samples} samples for testing")
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output") + 1
        if idx < len(sys.argv):
            output_dir = sys.argv[idx]
    
    print("=" * 60)
    print("MedRAG/textbooks Dataset Downloader")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    else:
        print("Downloading all samples...")
    print()
    
    download_medrag_textbooks(output_dir=output_dir, max_samples=max_samples)
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"\nData saved to: {os.path.abspath(output_dir)}/")
    print("  - textbooks_chunks.json (all chunks)")
    print("  - markdown/ (individual textbook files)")


if __name__ == "__main__":
    main()
