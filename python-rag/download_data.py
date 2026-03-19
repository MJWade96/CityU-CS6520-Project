"""
Download Medical Datasets Script

Downloads:
1. MedQA - USMLE questions (from existing benchmark directory)
2. PubMed Abstracts - Medical research abstracts

Usage:
    python download_data.py
"""

import os
import json
import random
from pathlib import Path


def download_medqa(output_dir: str = "./data/evaluation") -> None:
    """Use existing MedQA dataset from benchmark directory"""
    print("=" * 60)
    print("Using Existing MedQA Dataset")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Path to existing benchmark MedQA files
    benchmark_dir = os.path.join(
        "..", "benchmark", "MedQA", "data_clean", "questions", "US"
    )
    test_file = os.path.join(benchmark_dir, "test.jsonl")
    dev_file = os.path.join(benchmark_dir, "dev.jsonl")
    train_file = os.path.join(benchmark_dir, "train.jsonl")

    # Check if files exist
    if not os.path.exists(test_file):
        print(f"Error: MedQA test file not found at {test_file}")
        return

    print(f"Found MedQA dataset in: {benchmark_dir}")
    print(f"Test file: {test_file}")
    print(f"Dev file: {dev_file}")
    print(f"Train file: {train_file}")

    # Read and process the test file (main dataset)
    try:
        print("\nProcessing MedQA dataset...")

        # Read JSONL file
        data = []
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    # Process the item to match expected format
                    question = item.get("question", "")
                    options = item.get("options", {})

                    if isinstance(options, dict):
                        # Convert dict options to list
                        sorted_keys = sorted(options.keys())
                        option_list = [options[k] for k in sorted_keys]
                        answer = item.get("answer", "")
                        if not answer and "answer_idx" in item:
                            answer_idx = item["answer_idx"]
                            if 0 <= answer_idx < len(option_list):
                                answer = option_list[answer_idx]
                    else:
                        option_list = options if isinstance(options, list) else []
                        answer = item.get("answer", "")

                    data.append(
                        {
                            "question": question,
                            "options": option_list,
                            "answer": answer,
                            "answer_index": (
                                option_list.index(answer)
                                if answer in option_list
                                else 0
                            ),
                            "source": "medqa",
                        }
                    )

        print(f"Processed {len(data)} MedQA questions from test file")

        # Save full dataset
        output_file = os.path.join(output_dir, "medqa.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved to: {output_file}")

        # Split into dev and test sets if needed
        if len(data) > 0:
            dev_size = min(300, len(data) // 2)
            random.seed(42)
            random.shuffle(data)

            dev_data = data[:dev_size]
            test_data = (
                data[dev_size : dev_size + 500]
                if len(data) > dev_size + 500
                else data[dev_size:]
            )

            # Save dev set
            dev_output = os.path.join(output_dir, "medqa_dev.json")
            with open(dev_output, "w", encoding="utf-8") as f:
                json.dump(dev_data, f, ensure_ascii=False, indent=2)

            # Save test set
            test_output = os.path.join(output_dir, "medqa_test.json")
            with open(test_output, "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)

            print(f"Dev set: {len(dev_data)} questions -> {dev_output}")
            print(f"Test set: {len(test_data)} questions -> {test_output}")

    except Exception as e:
        print(f"Error processing MedQA dataset: {e}")
        import traceback

        traceback.print_exc()


def download_pubmed(
    output_dir: str = "./data/corpus/pubmed",
    max_results: int = 1000,
    skip: bool = False,
) -> None:
    """Download PubMed abstracts"""
    if skip:
        print("=" * 60)
        print("Skipping PubMed Download")
        print("=" * 60)
        print("Note: PubMed abstracts download skipped.")
        print("You can download them later by running:")
        print("  python download_data.py --with-pubmed")
        return

    print("=" * 60)
    print("Downloading PubMed Abstracts")
    print("=" * 60)

    try:
        from Bio import Entrez
    except ImportError:
        print("Error: Biopython not installed")
        print("Install with: pip install biopython")
        return

    Entrez.email = "researcher@example.com"

    queries = [
        "hypertension[Title/Abstract]",
        "diabetes[Title/Abstract]",
        "myocardial infarction[Title/Abstract]",
        "pneumonia[Title/Abstract]",
        "stroke[Title/Abstract]",
        "copd[Title/Abstract]",
        "depression[Title/Abstract]",
        "cancer[Title/Abstract]",
    ]

    all_records = []
    records_per_query = max_results // len(queries)

    for query in queries:
        print(f"\nSearching: {query}")
        try:
            handle = Entrez.esearch(
                db="pubmed", term=query, retmax=records_per_query, sort="relevance"
            )
            search_results = Entrez.read(handle)
            id_list = search_results["IdList"]

            if not id_list:
                print(f"  No results found for: {query}")
                continue

            from Bio import Medline

            for pmid in id_list[:50]:
                try:
                    handle = Entrez.efetch(
                        db="pubmed", id=pmid, rettype="medline", retmode="text"
                    )
                    records = Medline.parse(handle)
                    for record in records:
                        all_records.append(
                            {
                                "pmid": pmid,
                                "title": record.get("TI", ""),
                                "abstract": record.get("AB", ""),
                                "authors": record.get("AU", []),
                                "journal": record.get("JT", ""),
                                "year": (
                                    record.get("DP", "").split()[0]
                                    if record.get("DP")
                                    else ""
                                ),
                            }
                        )
                    handle.close()
                except Exception:
                    continue

            print(
                f"  Found {len(id_list)} results, processed {len(all_records)} so far..."
            )

        except Exception as e:
            print(f"  Error: {e}")
            continue

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pubmed_abstracts.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully downloaded {len(all_records)} PubMed abstracts")
    print(f"Saved to: {output_file}")


def create_statpearls_directories(base_dir: str = "./data/corpus") -> None:
    """Create StatPearls directory structure"""
    print("=" * 60)
    print("Creating StatPearls Directory")
    print("=" * 60)

    statpearls_dir = os.path.join(base_dir, "statpearls")
    os.makedirs(statpearls_dir, exist_ok=True)

    readme_content = """# StatPearls Medical Knowledge Base

This directory should contain StatPearls markdown files.

## How to obtain StatPearls data:

1. Visit https://www.statpearls.com/
2. Register for an account
3. Download the medical articles in Markdown format
4. Place the .md files in this directory

## Structure:
- Each article should be a separate .md file
- Articles will be automatically chunked (256-512 tokens) during loading

## Note:
StatPearls is a free, peer-reviewed medical knowledge base.
The data is publicly available but requires registration to download.
"""

    readme_path = os.path.join(statpearls_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"Created directory: {statpearls_dir}")
    print(f"Please download StatPearls data from https://www.statpearls.com/")


def create_data_directories(base_dir: str = "./data") -> None:
    """Create all required data directories"""
    dirs = [
        os.path.join(base_dir, "corpus", "statpearls"),
        os.path.join(base_dir, "corpus", "pubmed"),
        os.path.join(base_dir, "evaluation"),
        os.path.join(base_dir, "vector_store"),
        os.path.join(base_dir, "medical_knowledge"),
    ]

    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")


def main():
    """Main function - runs all downloads by default"""
    import sys

    skip_pubmed = "--skip-pubmed" in sys.argv or "--skip" in sys.argv
    output_dir = "./data"

    print("=" * 60)
    print("Medical Data Downloader")
    print("=" * 60)
    print("\nThis will use:")
    print("  1. Existing MedQA (USMLE questions) from benchmark directory")
    print("  2. Download PubMed abstracts" + (" [SKIPPED]" if skip_pubmed else ""))
    print("  3. Create StatPearls directory")
    print()

    create_data_directories(output_dir)

    print("\n" + "=" * 60)
    print("Step 1: Using Existing MedQA Dataset")
    print("=" * 60)
    download_medqa(os.path.join(output_dir, "evaluation"))

    print("\n" + "=" * 60)
    print("Step 2: Downloading PubMed Abstracts")
    print("=" * 60)
    download_pubmed(os.path.join(output_dir, "corpus", "pubmed"), skip=skip_pubmed)

    print("\n" + "=" * 60)
    print("Step 3: Creating StatPearls Directory")
    print("=" * 60)
    create_statpearls_directories(os.path.join(output_dir, "corpus"))

    print("\n" + "=" * 60)
    print("Process Complete!")
    print("=" * 60)
    print(f"\nData saved to: {output_dir}/")
    print("  - evaluation/medqa.json")
    print(
        "  - corpus/pubmed/pubmed_abstracts.json"
        + (" [SKIPPED]" if skip_pubmed else "")
    )
    print("  - corpus/statpearls/ (requires manual download)")

    if skip_pubmed:
        print("\nTo download PubMed data later, run:")
        print("  python download_data.py")


if __name__ == "__main__":
    main()
