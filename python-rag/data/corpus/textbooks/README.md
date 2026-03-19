# Medical Textbooks Corpus (MedRAG)

This directory contains the medical textbooks corpus from the MedRAG project.

## Data Source

**Source**: MedRAG/textbooks dataset from Hugging Face  
**URL**: https://huggingface.co/datasets/MedRAG/textbooks  
**Reference**: MedRAG paper - Benchmarking Retrieval-Augmented Generation for Medicine (Xiong et al., 2024)

## Contents

This corpus contains **17 medical textbooks** covering various medical disciplines:

### Basic Sciences
- **Anatomy_Gray** - Gray's Anatomy
- **Biochemistry_Lippincott** - Lippincott's Biochemistry
- **Cell_Biology_Alberts** - Alberts' Cell Biology
- **Histology_Ross** - Ross' Histology
- **Physiology_Levy** - Levy's Physiology
- **Immunology_Janeway** - Janeway's Immunology
- **Pathology_Robbins** - Robbins' Pathology
- **Pathoma_Husain** - Pathoma (Husain)

### Clinical Sciences
- **InternalMed_Harrison** - Harrison's Internal Medicine
- **Pharmacology_Katzung** - Katzung's Pharmacology
- **Neurology_Adams** - Adams' Neurology
- **Psychiatry_DSM-5** - DSM-5 Psychiatry
- **Pediatrics_Nelson** - Nelson's Pediatrics
- **Surgery_Schwartz** - Schwartz's Surgery
- **Obstetrics_Williams** - Williams' Obstetrics
- **Gynecology_Novak** - Novak's Gynecology

### Board Review
- **First_Aid_Step1** - First Aid for USMLE Step 1
- **First_Aid_Step2** - First Aid for USMLE Step 2

## Data Structure

### Raw Data (chunk/)
The `chunk/` directory contains 17 JSONL files, one per textbook. Each line is a JSON object with:
```json
{
  "id": "Anatomy_Gray_0",
  "title": "Anatomy_Gray",
  "content": "What is anatomy? Anatomy includes...",
  "contents": "Anatomy_Gray. What is anatomy?..."
}
```

### Processed Data
- **textbooks_combined.json**: All chunks combined into a single JSON file with standardized format
  - Added `source` field: "medrag_textbooks"
  - Added `textbook` field: textbook identifier

### Statistics
- **Total chunks**: ~126,000 (varies by textbook)
- **Average chunk length**: ~500-800 characters
- **Total size**: ~180 MB (raw JSONL files)

## How to Download

### Option 1: Using huggingface-cli (Recommended)
```bash
cd python-rag
huggingface-cli download MedRAG/textbooks --repo-type=dataset --local-dir data/corpus/textbooks
```

### Option 2: Using Python script
```bash
python download_textbooks.py
```

### Option 3: Manual download
1. Visit https://huggingface.co/datasets/MedRAG/textbooks
2. Download all files from the `chunk/` directory
3. Place them in `data/corpus/textbooks/chunk/`

## How to Process

After downloading, process the data:
```bash
python process_textbooks.py
```

This will:
- Combine all JSONL files
- Add metadata fields
- Save as `textbooks_combined.json`

## Usage in RAG System

The processed data can be used with the RAG system's document loader:

```python
from app.rag.document_loader import load_documents

# Load textbooks corpus
documents = load_documents("./data/corpus/textbooks/textbooks_combined.json")
```

## Citation

If you use this dataset, please cite the MedRAG paper:

```bibtex
@article{xiong2024benchmarking,
    title={Benchmarking Retrieval-Augmented Generation for Medicine},
    author={Guangzhi Xiong and Qiao Jin and Zhiyong Lu and Aidong Zhang},
    journal={arXiv preprint arXiv:2402.13178},
    year={2024}
}
```

## License

This dataset is for research and educational purposes. Please refer to the original textbooks' licenses for usage rights.

## Notes

- This corpus replaces the original StatPearls corpus planned in the project design
- MedRAG/textbooks provides better coverage and is more readily available
- The chunking is already done according to MedRAG's methodology (paragraph-level)
- Compatible with the RAG system's existing architecture
