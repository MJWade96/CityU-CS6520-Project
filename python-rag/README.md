# Medical RAG Evaluation Toolkit

This repository is centered on corpus preparation and evaluation workflows for the medical RAG experiments. The supported runtime surface is the CLI scripts under `python-rag/`.

## Overview

The project now uses a single native LlamaIndex retrieval stack behind the original entrypoint names:

- `build_vector_index.py`: build the FAISS-backed native index used by all RAG scripts.
- `complete_eval.py`: primary RAG evaluation using `HuggingFaceEmbedding`, `OpenAILike`, `FaissVectorStore`, and `VectorStoreIndex` query capability.
- `enhanced_eval.py`: enhanced RAG evaluation using native hybrid retrieval, query rewrite, and reranking.
- `sample_validation.py`: small no-RAG vs RAG comparison using the same native store.
- `evaluate_no_rag.py`: direct LLM baseline without retrieval.
- `run_with_resume.py`: restart supported evaluation scripts from checkpoints.

## Setup

```bash
cd python-rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Configure the LLM and embedding environment through the constants and environment variables used by `app/rag/evaluation/eval_shared.py` and `app/rag/retriever/runtime_config.py`.

The data root is resolved by `app/rag/data/data_paths.py`: if `python-rag/data/` exists it is used directly; otherwise the scripts fall back to the sibling directory `RAG_Medical_Data/` next to `python-rag/`, unless `RAG_DATA_DIR` is set.

## Common Workflow

```bash
# Optional data preparation
python download_statpearls.py
python combine_corpora.py

# Rebuild the native FAISS-backed index after pulling migration changes
python build_vector_index.py

# Quick sanity check before full runs
python sample_validation.py

# Main evaluations
python evaluate_no_rag.py
python complete_eval.py
```

## Script Prerequisites

- `build_vector_index.py` requires `<data-root>/corpus/combined_corpus.json` because it rebuilds the persisted index from the combined corpus.
- `complete_eval.py`, `enhanced_eval.py`, and `sample_validation.py` require `<data-root>/vector_store/faiss_index` plus `<data-root>/evaluation/medqa.json`.
- `evaluate_no_rag.py` only needs `<data-root>/evaluation/medqa.json` and valid LLM settings.

If a supported long evaluation is interrupted, run:

```bash
python run_with_resume.py
```

`run_with_resume.py` defaults to auto-detecting interrupted runs for `complete_eval.py`, `sample_validation.py`, and `evaluate_no_rag.py`.

## Active Code Paths

The current scripts share a small core set of modules:

- `app/rag/evaluation/eval_shared.py`: prompt building, answer extraction, concurrency, and API helpers.
- `app/rag/utils/progress_manager.py`: checkpoints and live/final artifacts.
- `app/rag/data/data_paths.py`: canonical dataset, cache, and output paths.
- `app/rag/data/corpus_loader.py`: shared combined-corpus parsing and metadata mapping.
- `app/rag/evaluation/no_rag_eval.py`: baseline evaluation flow.
- `app/rag/evaluation/naive_rag_eval.py`: native RAG evaluation flow behind the primary entrypoint names.
- `app/rag/evaluation/enhanced_rag_eval.py`: enhanced native RAG evaluation flow behind `enhanced_eval.py`.
- `app/rag/evaluation/sample_validation_eval.py`: sample-comparison implementation behind `sample_validation.py`.
- `app/rag/retriever/vector_store.py`: native FAISS-backed storage, retrieval, and query-engine helpers.
- `app/rag/retriever/runtime_config.py`: embedding model and device resolution.

## Project Structure

```text
python-rag/
├── app/
│   ├── __init__.py
│   └── rag/
│       ├── data/
│       ├── evaluation/
│       ├── retriever/
│       └── utils/
├── build_vector_index.py
├── combine_corpora.py
├── complete_eval.py
├── download_statpearls.py
├── enhanced_eval.py
├── evaluate_no_rag.py
├── run_with_resume.py
├── sample_validation.py
└── README.md
```

## Notes

1. Evaluation artifacts are written under `results/evaluation/`.
2. The on-disk index format behind `<data-root>/vector_store/faiss_index` is now the native LlamaIndex-backed format, so rebuild it with `python build_vector_index.py` after pulling this migration.
3. Corpus-preparation scripts and evaluation scripts do not share identical prerequisites; only index builders need `combined_corpus.json`.
