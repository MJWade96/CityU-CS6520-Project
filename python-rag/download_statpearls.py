"""
Download StatPearls Dataset from Hugging Face

Downloads the MedRAG/StatPearls dataset and processes it for RAG system.

Usage:
    python download_statpearls.py
"""

import os
import json
from pathlib import Path


def download_statpearls_huggingface(
    output_dir: str = "./data/corpus/statpearls",
    max_samples: int = None,
) -> None:
    """
    Download StatPearls dataset from Hugging Face
    
    Args:
        output_dir: Directory to save the processed data
        max_samples: Maximum number of samples to download (None for all)
    """
    print("=" * 60)
    print("Downloading StatPearls Dataset from Hugging Face")
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
    print("\nLoading MedRAG/StatPearls dataset...")
    try:
        dataset = load_dataset("MedRAG/statpearls")
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        print("\nTrying alternative approach...")
        # Try loading without split
        try:
            dataset = load_dataset("MedRAG/statpearls", split="train")
            dataset = {"train": dataset}
            print(f"Dataset loaded successfully!")
        except Exception as e2:
            print(f"Error: {e2}")
            import traceback
            traceback.print_exc()
            return
    
    # Process the dataset
    all_articles = []
    
    for split_name, split_data in dataset.items():
        print(f"\nProcessing {split_name} split...")
        
        for i, item in enumerate(split_data):
            if max_samples and i >= max_samples:
                break
                
            article = {
                "id": item.get("id", f"{split_name}_{i}"),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "source": "statpearls",
                "split": split_name,
            }
            
            all_articles.append(article)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} articles...")
    
    print(f"\nTotal articles processed: {len(all_articles)}")
    
    # Save as JSON
    output_file = os.path.join(output_dir, "statpearls_articles.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to: {output_file}")
    
    # Also save each article as separate markdown file
    markdown_dir = os.path.join(output_dir, "markdown")
    os.makedirs(markdown_dir, exist_ok=True)
    
    print(f"\nSaving individual markdown files to: {markdown_dir}")
    
    for i, article in enumerate(all_articles):
        md_content = f"# {article['title']}\n\n"
        md_content += f"**Source:** StatPearls\n"
        md_content += f"**ID:** {article['id']}\n\n"
        md_content += f"{article['content']}\n"
        
        # Create filename from title (sanitize)
        safe_title = "".join(
            c for c in article["title"] if c.isalnum() or c in " -_"
        ).strip()
        safe_title = safe_title.replace(" ", "_")[:50]  # Limit length
        
        md_file = os.path.join(markdown_dir, f"{safe_title}_{i}.md")
        
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        if (i + 1) % 100 == 0:
            print(f"  Saved {i + 1} markdown files...")
    
    print(f"\nSuccessfully saved {len(all_articles)} articles")
    print(f"JSON format: {output_file}")
    print(f"Markdown files: {markdown_dir}/")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total articles: {len(all_articles)}")
    
    # Calculate average length
    total_chars = sum(len(article["content"]) for article in all_articles)
    avg_chars = total_chars // len(all_articles) if all_articles else 0
    print(f"Average content length: {avg_chars} characters")
    print(f"Total characters: {total_chars:,}")


def main():
    """Main function"""
    import sys
    
    output_dir = "./data/corpus/statpearls"
    max_samples = None
    
    # Parse command line arguments
    if "--max-samples" in sys.argv:
        idx = sys.argv.index("--max-samples") + 1
        if idx < len(sys.argv):
            max_samples = int(sys.argv[idx])
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output") + 1
        if idx < len(sys.argv):
            output_dir = sys.argv[idx]
    
    print("=" * 60)
    print("StatPearls Dataset Downloader")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    else:
        print("Downloading all samples...")
    
    download_statpearls_huggingface(output_dir=output_dir, max_samples=max_samples)
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
