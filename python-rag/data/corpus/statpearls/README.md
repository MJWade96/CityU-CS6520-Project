# Medical Knowledge Corpus - Textbooks (MedRAG)

## Important Note

**This directory now uses MedRAG/textbooks instead of StatPearls**

Due to privacy policy restrictions on StatPearls content distribution, we have adopted the **MedRAG/textbooks** corpus as our primary medical knowledge base.

## Why Textbooks Instead of StatPearls?

1. **Accessibility**: MedRAG/textbooks is freely available on Hugging Face without registration
2. **Quality**: Contains 17 authoritative medical textbooks covering basic and clinical sciences
3. **Pre-processed**: Already chunked according to MedRAG methodology
4. **Compatibility**: Same format as StatPearls, works with existing RAG pipeline

## Data Location

The actual data is stored in:
```
./data/corpus/textbooks/
```

This directory serves as a compatibility layer for the RAG system.

## Textbooks Included

### Basic Sciences (8 textbooks)
- Gray's Anatomy
- Lippincott's Biochemistry
- Alberts' Cell Biology
- Ross' Histology
- Levy's Physiology
- Janeway's Immunology
- Robbins' Pathology
- Pathoma (Husain)

### Clinical Sciences (7 textbooks)
- Harrison's Internal Medicine
- Katzung's Pharmacology
- Adams' Neurology
- DSM-5 Psychiatry
- Nelson's Pediatrics
- Schwartz's Surgery
- Williams' Obstetrics & Novak's Gynecology

### Board Review (2 textbooks)
- First Aid for USMLE Step 1
- First Aid for USMLE Step 2

## Statistics

- **Total chunks**: ~126,000
- **Average chunk length**: ~500-800 characters
- **Total size**: ~180 MB

## Usage

The RAG system will automatically use the textbooks corpus. No configuration changes needed.

For more details, see:
- `../textbooks/README.md` - Full documentation
- `../textbooks/chunk/` - Raw JSONL files
- `../textbooks/textbooks_combined.json` - Processed combined file

## Reference

MedRAG Paper: [Benchmarking Retrieval-Augmented Generation for Medicine](https://arxiv.org/abs/2402.13178)

Hugging Face Dataset: [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks)
