"""
Download PubMed Abstracts for Medical RAG System

Reference: RAG_Mistral project (https://github.com/mehdiir/RAG_Mistral)
Uses BioPython + Entrez API to download recent PubMed articles

This script downloads:
- ~1000 PubMed abstracts related to common medical conditions
- Saves as JSON format compatible with the RAG system
"""

import os
import json
import time
from pathlib import Path
from Bio import Entrez, Medline


def setup_email():
    """Setup Entrez email (required by NCBI)"""
    # NCBI requires an email address for API access
    email = "researcher@example.com"  # Replace with your email
    Entrez.email = email
    return email


def search_pubmed(query, max_results=100):
    """
    Search PubMed and return list of PMIDs
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
    
    Returns:
        List of PubMed IDs
    """
    print(f"  Searching: {query}")
    
    try:
        # Search PubMed
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance",
            usehistory="y"
        )
        search_results = Entrez.read(handle)
        handle.close()
        
        id_list = search_results["IdList"]
        print(f"    Found {len(id_list)} results")
        
        return id_list
    
    except Exception as e:
        print(f"    Error searching: {e}")
        return []


def fetch_abstracts(pmid_list, batch_size=50):
    """
    Fetch abstract details from PubMed
    
    Args:
        pmid_list: List of PubMed IDs
        batch_size: Number of abstracts to fetch per request
    
    Returns:
        List of abstract dictionaries
    """
    abstracts = []
    
    # Process in batches to avoid API rate limits
    for i in range(0, len(pmid_list), batch_size):
        batch = pmid_list[i:i + batch_size]
        
        try:
            # Fetch abstracts
            handle = Entrez.efetch(
                db="pubmed",
                id=batch,
                rettype="medline",
                retmode="text"
            )
            
            # Parse MEDLINE format
            records = Medline.parse(handle)
            
            for record in records:
                abstract = {
                    "pmid": record.get("PMID", ""),
                    "title": record.get("TI", ""),
                    "abstract": record.get("AB", ""),
                    "authors": record.get("AU", []),
                    "journal": record.get("JT", ""),
                    "year": record.get("DP", "").split()[0] if record.get("DP") else "",
                    "source": "pubmed"
                }
                
                # Only include if abstract exists
                if abstract["abstract"]:
                    abstracts.append(abstract)
            
            handle.close()
            print(f"    Fetched {len(abstracts)} abstracts so far...")
            
            # Respect NCBI rate limits (3 requests per second max)
            time.sleep(0.5)
        
        except Exception as e:
            print(f"    Error fetching batch: {e}")
            time.sleep(1)  # Wait longer on error
    
    return abstracts


def download_pubmed_abstracts(
    output_dir: str = "./data/corpus/pubmed",
    total_target: int = 1000,
):
    """
    Main function to download PubMed abstracts
    
    Args:
        output_dir: Directory to save downloaded data
        total_target: Target number of abstracts to download
    """
    print("=" * 60)
    print("Downloading PubMed Abstracts")
    print("=" * 60)
    print(f"Target: {total_target} abstracts")
    print(f"Output: {output_dir}")
    print()
    
    # Setup
    email = setup_email()
    print(f"Using email: {email}")
    print("Note: NCBI requires email for API access")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define medical topics (similar to RAG_Mistral approach)
    medical_topics = [
        # Common diseases
        "hypertension[Title/Abstract]",
        "diabetes mellitus[Title/Abstract]",
        "myocardial infarction[Title/Abstract]",
        "stroke[Title/Abstract]",
        "pneumonia[Title/Abstract]",
        "copd[Title/Abstract]",
        "depression[Title/Abstract]",
        "cancer[Title/Abstract]",
        
        # Symptoms and conditions
        "chest pain[Title/Abstract]",
        "headache[Title/Abstract]",
        "abdominal pain[Title/Abstract]",
        "fever[Title/Abstract]",
        "dyspnea[Title/Abstract]",
        "fatigue[Title/Abstract]",
        
        # Treatments
        "antihypertensive therapy[Title/Abstract]",
        "chemotherapy[Title/Abstract]",
        "surgery[Title/Abstract]",
        "radiotherapy[Title/Abstract]",
        
        # Diagnostics
        "diagnosis[Title/Abstract]",
        "biomarker[Title/Abstract]",
        "imaging[Title/Abstract]",
        "genetic testing[Title/Abstract]",
    ]
    
    # Calculate abstracts per topic
    abstracts_per_topic = total_target // len(medical_topics)
    
    all_abstracts = []
    
    # Download abstracts for each topic
    for i, topic in enumerate(medical_topics, 1):
        print(f"\n[{i}/{len(medical_topics)}] Processing topic:")
        
        # Search
        pmids = search_pubmed(topic, max_results=abstracts_per_topic + 50)
        
        if not pmids:
            print("  No results, skipping...")
            continue
        
        # Fetch abstracts
        abstracts = fetch_abstracts(pmids[:abstracts_per_topic])
        
        print(f"  ✓ Downloaded {len(abstracts)} abstracts")
        
        all_abstracts.extend(abstracts)
        
        # Progress report
        print(f"  Total so far: {len(all_abstracts)} abstracts")
        
        # Be nice to NCBI servers
        time.sleep(1)
    
    # Remove duplicates (by PMID)
    seen_pmids = set()
    unique_abstracts = []
    for abstract in all_abstracts:
        if abstract["pmid"] not in seen_pmids:
            seen_pmids.add(abstract["pmid"])
            unique_abstracts.append(abstract)
    
    print(f"\n{'=' * 60}")
    print(f"Download Complete!")
    print(f"Total abstracts: {len(unique_abstracts)}")
    print(f"Unique abstracts: {len(unique_abstracts)}")
    
    # Save to JSON
    output_file = os.path.join(output_dir, "pubmed_abstracts.json")
    
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_abstracts, f, ensure_ascii=False, indent=2)
    
    file_size = os.path.getsize(output_file)
    print(f"✓ Saved ({file_size / (1024*1024):.1f} MB)")
    
    # Statistics
    print(f"\n{'=' * 60}")
    print("Statistics:")
    
    # Abstracts with full text
    with_abstract = sum(1 for a in unique_abstracts if a["abstract"])
    print(f"  Abstracts with content: {with_abstract}")
    
    # Average length
    avg_length = sum(len(a["abstract"]) for a in unique_abstracts if a["abstract"]) // len(unique_abstracts)
    print(f"  Average abstract length: {avg_length:,} characters")
    
    # Years
    years = [a["year"] for a in unique_abstracts if a["year"]]
    if years:
        print(f"  Year range: {min(years)} - {max(years)}")
    
    # Top journals
    journal_counts = {}
    for a in unique_abstracts:
        journal = a["journal"]
        if journal:
            journal_counts[journal] = journal_counts.get(journal, 0) + 1
    
    print(f"\n  Top journals:")
    for journal, count in sorted(journal_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"    {journal}: {count}")
    
    print(f"\n{'=' * 60}")
    print("SUCCESS!")
    print(f"{'=' * 60}")
    
    return len(unique_abstracts)


def main():
    """Main entry point"""
    import sys
    
    # Default settings
    output_dir = "./data/corpus/pubmed"
    total_target = 1000
    
    # Parse command line arguments
    if "--output" in sys.argv:
        idx = sys.argv.index("--output") + 1
        if idx < len(sys.argv):
            output_dir = sys.argv[idx]
    
    if "--count" in sys.argv:
        idx = sys.argv.index("--count") + 1
        if idx < len(sys.argv):
            total_target = int(sys.argv[idx])
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python download_pubmed.py [options]")
        print("\nOptions:")
        print("  --output DIR    Output directory (default: ./data/corpus/pubmed)")
        print("  --count N       Number of abstracts to download (default: 1000)")
        print("  --help, -h      Show this help message")
        return
    
    # Download
    count = download_pubmed_abstracts(
        output_dir=output_dir,
        total_target=total_target
    )
    
    print(f"\nDownloaded {count} PubMed abstracts")
    print(f"Data saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
