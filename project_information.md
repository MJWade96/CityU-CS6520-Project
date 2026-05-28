# Project Information

## 2026-05-28

- Fact: Project stage 1 covers corpus expansion and ablation experiments, with GraphRAG, RAGAS, transfer evaluation, and FAISS Flat/IVF/HNSW ablation left out of this implementation stage.
  Evidence: `docs/project information.md`; verified from project documentation.

- Fact: Phase 1 uses small-sample smoke runs to prove the corpus/index/evaluation chain is executable; smoke accuracy is not a formal ablation result.
  Evidence: `run_phase1_ablation.py`; verified from implementation.

- Fact: MedRAG Textbooks is the TextBook corpus source for phase 1. The local sync entrypoint is `app/rag/data/medical_corpus/download_textbooks.py`, and normalized records are written to `app/rag/data/medical_corpus/textbooks_corpus.json`. This is a project-local MedRAG-compatible artifact, not a MedScore-native file.
  Evidence: `app/rag/data/textbooks_dataset.py`; verified from implementation and MedScore/MedRAG documentation.

- Fact: Registered corpora are normalized through `app/rag/data/corpus_registry.py` before merging. The active phase 1 sources are `statpearls` and `textbooks`. StatPearls uses the existing `app/rag/data/medical_corpus/statpearls.py` processing path and the already-downloaded `statpearls_NBK430685` source directory; phase 1 reads the existing `combined_corpus.json` as the StatPearls artifact.
  Evidence: `app/rag/data/corpus_registry.py`, `app/rag/data/medical_corpus/statpearls.py`; verified from implementation and user correction.

- Fact: Phase 1 index builds use `IndexBuildConfig` so corpus file, embedding model, FAISS index type, corpus version, and build timing are recorded in index metadata.
  Evidence: `app/rag/data/medical_corpus/build_vector_index.py`; verified from implementation.

- Fact: Enhanced RAG now separates retrieval candidate count from reranker output count. `retrieval_top_k` controls candidates sent into retrieval/reranking, `reranker_top_k` controls final reranker output, and `hybrid_alpha` is the dense weight with BM25 weight equal to `1 - alpha`.
  Evidence: `app/rag/evaluation/enhanced_rag_eval.py`; verified from implementation and LlamaIndex node postprocessor docs.

- Fact: Textbooks was successfully synced from `MedRAG/textbooks` into `app/rag/data/medical_corpus/textbooks_corpus.json` with 125,847 records.
  Evidence: `python app\rag\data\medical_corpus\download_textbooks.py` runtime output; verified from runtime output and generated file inspection.

- Fact: Phase 1 smoke run completed with `sample_size=5`, corpus variants `statpearls` and `statpearls_textbooks`, embedding `BAAI/bge-m3`, and k values `3/5/10`. It generated small ignored indexes and wrote `results/evaluation/phase1/phase1_smoke_summary.json` plus `.csv`. These are pipeline verification artifacts, not formal ablation results.
  Evidence: `python run_phase1_ablation.py` runtime output and `results/evaluation/phase1/phase1_smoke_summary.json`; verified from runtime output and generated summary inspection.
