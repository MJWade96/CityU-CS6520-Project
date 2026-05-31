# Medical RAG Evaluation Toolkit

This repository is centered on corpus preparation and evaluation workflows for the medical RAG experiments. Experiment orchestration entrypoints live under `app/rag/experiments/`, matching the architecture boundary in `docs/architecture.md`.

## Overview

The project now uses a single native LlamaIndex retrieval stack behind the experiment entrypoints:

- `app/rag/data/medical_corpus/build_vector_index.py`: build the FAISS-backed native index used by RAG evaluations.
- `app/rag/experiments/complete_eval.py`: primary RAG evaluation using API embeddings, `OpenAILike`, `FaissVectorStore`, and `VectorStoreIndex` query capability.
- `app/rag/experiments/enhanced_eval.py`: enhanced RAG evaluation using native hybrid retrieval, query rewrite, and reranking.
- `app/rag/experiments/sample_validation.py`: small no-RAG vs RAG comparison using the same native store.
- `app/rag/experiments/evaluate_no_rag.py`: direct LLM baseline without retrieval.
- `app/rag/experiments/run_phase1_ablation.py`: phase 1 smoke runner for corpus and retrieval ablation wiring.
- `app/rag/experiments/run_formal_ablation.py`: formal phase 1 ablation framework entrypoint; by default it writes the framework, matrix, and cache manifest only.
- `app/rag/experiments/run_with_resume.py`: restart supported evaluation entrypoints from checkpoints.

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
python -m app.rag.data.medical_corpus.download_statpearls
python -m app.rag.data.medical_corpus.combine_corpora

# Rebuild the native FAISS-backed index after pulling migration changes
python -m app.rag.data.medical_corpus.build_vector_index

# Quick sanity check before full runs
python -m app.rag.experiments.sample_validation

# Main evaluations
python -m app.rag.experiments.evaluate_no_rag
python -m app.rag.experiments.complete_eval

# Formal phase 1 framework artifacts only; this does not run accuracy evaluation
python -m app.rag.experiments.run_formal_ablation
```

## Formal Phase 1 Ablation

`python -m app.rag.experiments.run_formal_ablation` writes the formal framework JSON, matrix CSV, and cache manifest under `results/runs/`. The default `FORMAL_RUN_IDS_TO_EXECUTE` is empty, so the command reports `Executed formal runs: 0` and does not run the formal accuracy jobs.

Only fully resolved formal rows should be executed. Rows that still depend on earlier-stage winners, such as `stage1_top...`, `stage2_top...`, `best_k`, or `best_alpha`, are rejected by the formal runtime until those concrete selections are resolved.

## Script Prerequisites

- `app/rag/data/medical_corpus/build_vector_index.py` requires `<data-root>/corpus/combined_corpus.json` because it rebuilds the persisted index from the combined corpus.
- `app/rag/experiments/complete_eval.py`, `app/rag/experiments/enhanced_eval.py`, and `app/rag/experiments/sample_validation.py` require `<data-root>/vector_store/faiss_index` plus `<data-root>/evaluation/medqa.json`.
- `app/rag/experiments/evaluate_no_rag.py` only needs `<data-root>/evaluation/medqa.json` and valid LLM settings.

If a supported long evaluation is interrupted, run:

```bash
python -m app.rag.experiments.run_with_resume
```

`app/rag/experiments/run_with_resume.py` defaults to auto-detecting interrupted runs for `complete_eval.py`, `sample_validation.py`, and `evaluate_no_rag.py`.

## Active Code Paths

The current scripts share a small core set of modules:

- `app/rag/evaluation/eval_shared.py`: prompt building, answer extraction, concurrency, and API helpers.
- `app/rag/utils/progress_manager.py`: checkpoints and live/final artifacts.
- `app/rag/data/data_paths.py`: canonical dataset, cache, and output paths.
- `app/rag/data/corpus_loader.py`: shared combined-corpus parsing and metadata mapping.
- `app/rag/evaluation/no_rag_eval.py`: baseline evaluation flow.
- `app/rag/evaluation/naive_rag_eval.py`: native RAG evaluation flow behind the primary entrypoint names.
- `app/rag/evaluation/enhanced_rag_eval.py`: enhanced native RAG evaluation flow behind `app/rag/experiments/enhanced_eval.py`.
- `app/rag/evaluation/sample_validation_eval.py`: sample-comparison implementation behind `app/rag/experiments/sample_validation.py`.
- `app/rag/retriever/vector_store.py`: native FAISS-backed storage, retrieval, and query-engine helpers.
- `app/rag/retriever/runtime_config.py`: API embedding model and endpoint resolution.

## Project Structure

```text
python-rag/
├── app/
│   ├── __init__.py
│   └── rag/
│       ├── data/
│       ├── evaluation/
│       ├── experiments/
│       ├── retriever/
│       └── utils/
├── results/
└── README.md
```

## Notes

1. Evaluation artifacts are written under `results/evaluation/`.
2. The on-disk index format behind `<data-root>/vector_store/faiss_index` is now the native LlamaIndex-backed format, so rebuild it with `python -m app.rag.data.medical_corpus.build_vector_index` after pulling this migration.
3. Corpus-preparation scripts and evaluation scripts do not share identical prerequisites; only index builders need `combined_corpus.json`.
