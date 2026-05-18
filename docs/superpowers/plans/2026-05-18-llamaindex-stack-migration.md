# LlamaIndex Stack Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mixed retrieval stack with a single native LlamaIndex implementation behind the original entrypoint and module names.

**Architecture:** Keep `build_vector_index.py`, `complete_eval.py`, and `sample_validation.py` as the public entrypoints. Implement their backing store and evaluation logic natively with LlamaIndex, remove parallel `llamaindex_*` files, and delete deprecated LangChain-only modules and dependencies.

**Tech Stack:** Python, LlamaIndex, FAISS, HuggingFace embeddings, OpenAI-compatible LLMs via `OpenAILike`, and focused pytest/py_compile/import validation.

---

### Task 1: Consolidate the primary module names onto native implementations

**Files:**
- Modify: `python-rag/app/rag/retriever/vector_store.py`
- Modify: `python-rag/app/rag/evaluation/naive_rag_eval.py`
- Modify: `python-rag/build_vector_index.py`
- Modify: `python-rag/complete_eval.py`
- Modify: `python-rag/sample_validation.py`
- Modify: `python-rag/requirements.txt`

- [x] **Step 1: Keep only the native dependency set**

Retain the minimal packages needed for `HuggingFaceEmbedding`, `OpenAILike`, FAISS-backed index persistence, and the direct OpenAI SDK used by shared helpers.

- [x] **Step 2: Move native implementations under the canonical module names**

Replace the contents of `vector_store.py`, `naive_rag_eval.py`, and `build_vector_index.py` with native LlamaIndex implementations so the old import paths remain the only supported ones.

- [x] **Step 3: Keep helper flows compatible with the new store**

Preserve the sample/staged helper scripts by exposing a native `similarity_search_with_score(...)` compatibility surface from the new store.

- [x] **Step 4: Keep shared runtime/config helpers neutral**

Keep `runtime_config.py`, `corpus_loader.py`, `config.py`, and `eval_shared.py` free of deprecated retrieval-stack dependencies.

### Task 2: Remove the parallel path and deprecated modules

**Files:**
- Delete: `python-rag/build_llamaindex_index.py`
- Delete: `python-rag/llamaindex_eval.py`
- Delete: `python-rag/llamaindex_sample_validation.py`
- Delete: `python-rag/enhanced_eval.py`
- Delete: `python-rag/app/rag/evaluation/llamaindex_rag_eval.py`
- Delete: `python-rag/app/rag/retriever/llamaindex_store.py`
- Delete: `python-rag/app/rag/retriever/embeddings.py`
- Delete: `python-rag/app/rag/retriever/hybrid_retriever.py`
- Delete: `python-rag/app/rag/retriever/query_rewrite.py`
- Delete: `python-rag/app/rag/retriever/reranker.py`
- Modify: `python-rag/run_with_resume.py`

- [x] **Step 1: Remove the parallel `llamaindex_*` entrypoints and modules**

Delete the secondary file set so the project only exposes the canonical old names.

- [x] **Step 2: Delete the deprecated LangChain-only helper modules**

Remove the hybrid/query-rewrite/reranker branch and the old embedding adapter layer so runtime Python files are free of `langchain` imports.

- [x] **Step 3: Narrow resume-helper support to retained scripts**

Update `run_with_resume.py` to track `complete_eval.py`, `sample_validation.py`, and `evaluate_no_rag.py` only.

### Task 3: Update docs and validate the final replacement state

**Files:**
- Modify: `python-rag/README.md`
- Modify: `docs/superpowers/specs/2026-05-18-llamaindex-stack-migration-design.md`
- Modify: `docs/superpowers/plans/2026-05-18-llamaindex-stack-migration.md`
- Verify: `python-rag/tests/test_llamaindex_native_stack.py`

- [x] **Step 1: Rewrite docs around the single native stack**

Document that `faiss_index` is now the single native store location, that the old primary entrypoint names remain in place, and that users must rebuild the index after pulling the migration.

- [x] **Step 2: Run architecture guardrail tests**

Run: `python -m pytest python-rag/tests/test_llamaindex_native_stack.py -q`

Expected: command exits with code 0.

- [ ] **Step 3: Run narrow syntax/import checks for retained entrypoints**

Run:

`python -m py_compile python-rag/build_vector_index.py`

`python -m py_compile python-rag/complete_eval.py`

`python -m py_compile python-rag/sample_validation.py`

`python -c "import pathlib, sys; sys.path.insert(0, str(pathlib.Path('python-rag').resolve())); import build_vector_index, complete_eval, sample_validation"`

Expected: all commands exit with code 0.

