# Medical RAG Evaluation Toolkit

This repository is now centered on evaluation and corpus-preparation workflows for the medical RAG experiments. The active runtime surface is the set of CLI scripts under the project root, not a FastAPI service.

## Overview

The project evaluates three main paths on the MedQA-style dataset:

- `evaluate_no_rag.py`: direct LLM baseline without retrieval.
- `complete_eval.py`: naive RAG evaluation with FAISS retrieval.
- `enhanced_eval.py`: hybrid retrieval + query rewrite + cross-encoder reranking.

There are also smaller utility entrypoints for validation and staged runs:

- `sample_validation.py`: small no-RAG vs naive-RAG comparison.
- `naive_rag_sample_eval.py`: sample-only naive-RAG validation.
- `naive_rag_retrieval.py`: cache retrieval results for the sample workflow.
- `naive_rag_generation.py`: generate answers from cached retrieval results.
- `run_with_resume.py`: restart supported evaluation scripts from checkpoints.

## Setup

```bash
cd python-rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Configure the LLM and embedding environment through the constants and environment variables already used by the scripts in `app/rag/eval_shared.py`, `app/rag/embeddings.py`, and `enhanced_eval.py`.

## Common Workflow

```bash
# Optional data preparation
python download_statpearls.py
python combine_corpora.py
python build_vector_index.py

# Quick sanity check before full runs
python sample_validation.py

# Main evaluations
python evaluate_no_rag.py
python complete_eval.py
python enhanced_eval.py
```

If a long evaluation is interrupted, run:

```bash
python run_with_resume.py
```

## Active Code Paths

Current evaluation entrypoints share a small core set of modules:

- `app/rag/eval_shared.py`: prompt building, answer extraction, concurrency, API helpers.
- `app/rag/progress_manager.py`: checkpoints and live/final artifacts.
- `app/rag/data_paths.py`: canonical dataset, cache, and output paths.
- `app/rag/no_rag_eval.py`: baseline evaluation flow.
- `app/rag/naive_rag_eval.py`: naive RAG evaluation flow.
- `app/rag/hybrid_retriever.py`, `app/rag/query_rewrite.py`, `app/rag/reranker.py`: enhanced evaluation stack.
- `app/rag/vector_store.py` and `app/rag/embeddings.py`: FAISS and embedding runtime support.

## Project Structure

```text
python-rag/
├── app/
│   ├── __init__.py
│   └── rag/
│       ├── __init__.py
│       ├── data_paths.py
│       ├── embeddings.py
│       ├── eval_shared.py
│       ├── hybrid_retriever.py
│       ├── json_utils.py
│       ├── naive_rag_eval.py
│       ├── no_rag_eval.py
│       ├── progress_manager.py
│       ├── query_rewrite.py
│       ├── reranker.py
│       └── vector_store.py
├── build_vector_index.py
├── combine_corpora.py
├── complete_eval.py
├── download_statpearls.py
├── enhanced_eval.py
├── evaluate_no_rag.py
├── naive_rag_generation.py
├── naive_rag_retrieval.py
├── naive_rag_sample_eval.py
├── run_with_resume.py
├── sample_validation.py
└── README.md
```

## Notes

1. The repository stores evaluation artifacts under `results/evaluation/`.
2. `run_with_resume.py` only manages the supported evaluation scripts it knows about.
3. The codebase keeps corpus-preparation scripts because rebuilding the FAISS index still depends on them.
