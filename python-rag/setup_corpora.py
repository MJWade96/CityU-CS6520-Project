"""
Update the corpus structure to use textbooks instead of StatPearls

Move the processed textbooks data to the statpearls directory structure
since the RAG system expects the data in that location.
"""

import os
import shutil
import json

def setup_corpora():
    """Setup corpora structure with textbooks replacing statpearls"""
    print("Setting up medical corpora...")
    
    # Source: processed textbooks
    textbooks_json = "./data/corpus/textbooks/textbooks_combined.json"
    
    # Destination: statpearls directory (as expected by RAG system)
    statpearls_dir = "./data/corpus/statpearls"
    os.makedirs(statpearls_dir, exist_ok=True)
    
    # Copy textbooks data to statpearls location
    dest_file = os.path.join(statpearls_dir, "statpearls_articles.json")
    
    if os.path.exists(textbooks_json):
        # Load textbooks data
        with open(textbooks_json, 'r', encoding='utf-8') as f:
            textbooks_data = json.load(f)
        
        # Update source field to reflect the actual source
        for chunk in textbooks_data:
            chunk["source"] = "medrag_textbooks"
        
        # Save to destination
        with open(dest_file, 'w', encoding='utf-8') as f:
            json.dump(textbooks_data, f, ensure_ascii=False, indent=2)
        
        print(f"Copied {len(textbooks_data):,} chunks to statpearls directory")
        print(f"Updated source field to 'medrag_textbooks'")
    else:
        print("Warning: textbooks_combined.json not found")
        print("Make sure to run the processing script first")
    
    # Also copy to the main corpus directory for consistency
    main_corpus_dir = "../data/corpus/statpearls"  # Relative to python-rag
    if os.path.exists("../data/corpus"):
        os.makedirs(main_corpus_dir, exist_ok=True)
        main_dest = os.path.join(main_corpus_dir, "statpearls_articles.json")
        if os.path.exists(textbooks_json):
            shutil.copy2(textbooks_json, main_dest)
            # Update source in copied file too
            with open(main_dest, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                for chunk in data:
                    chunk["source"] = "medrag_textbooks"
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.truncate()
    
    print("Corpus setup completed!")

if __name__ == "__main__":
    setup_corpora()