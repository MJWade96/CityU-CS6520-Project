# LlamaIndex Stack Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the enhanced evaluation feature surface while replacing its LangChain-bound retrieval stack with native LlamaIndex abstractions.

**Architecture:** Keep `build_vector_index.py`, `complete_eval.py`, `sample_validation.py`, and `enhanced_eval.py` as the public entrypoints. Reuse the current native store and evaluation helpers, restore the enhanced retrieval modules on native LlamaIndex primitives, and keep the recovered `ENHANCED_RAG` artifact naming contract intact.

**Tech Stack:** Python, LlamaIndex, FAISS, HuggingFace embeddings, OpenAI-compatible LLMs via `OpenAILike`, `BM25Retriever`, `QueryFusionRetriever`, native query transforms, `SentenceTransformerRerank`, and focused pytest/py_compile/import validation.

---

### Task 1: Recover the deleted enhanced contract and freeze scope

**Files:**
- Verify: `python-rag/enhanced_eval.py`
- Verify: `python-rag/app/rag/retriever/hybrid_retriever.py`
- Verify: `python-rag/app/rag/retriever/query_rewrite.py`
- Verify: `python-rag/app/rag/retriever/reranker.py`

- [x] **Step 1: Recover the last pre-deletion snapshot from git history**

Resolved the deleted enhanced files from commit `a878785`, which is the last snapshot before `Replace LangChain stack with native LlamaIndex paths` removed them.

- [x] **Step 2: Capture the user-visible contract that must survive migration**

Recovered these contract markers from history: artifact prefix `enhanced_rag_eval`, run name and evaluation type `ENHANCED_RAG`, checkpoint script names `enhanced_eval_dev` and `enhanced_eval_test`, plus the enhanced stages of hybrid retrieval, query rewrite, and reranking.

- [x] **Step 3: Freeze the implementation boundary before further coding**

Treat `enhanced_eval.py` as a preserved feature entrypoint and treat `hybrid_retriever.py`, `query_rewrite.py`, and `reranker.py` as implementation modules to migrate rather than delete.

### Task 2: Restore native enhanced module boundaries

**Files:**
- Add: `python-rag/enhanced_eval.py`
- Add: `python-rag/app/rag/retriever/hybrid_retriever.py`
- Add: `python-rag/app/rag/retriever/query_rewrite.py`
- Add: `python-rag/app/rag/retriever/reranker.py`
- Modify: `python-rag/requirements.txt`

- [x] **Step 1: Restore the deleted module surfaces with native imports**

Restored `enhanced_eval.py`, `hybrid_retriever.py`, `query_rewrite.py`, and `reranker.py` as native LlamaIndex modules so the enhanced surface exists again.

- [x] **Step 2: Rebuild hybrid retrieval around native LlamaIndex primitives**

`hybrid_retriever.py` now targets `QueryFusionRetriever` for fusion and the official `BM25Retriever` extension for sparse retrieval.

- [x] **Step 3: Rebuild query rewrite and reranking boundaries natively**

`query_rewrite.py` now exposes a native `BaseQueryTransform` boundary plus deterministic medical rewrite rules and an `OpenAILike`-backed rewriter, while `reranker.py` targets `SentenceTransformerRerank` through a Node Postprocessor boundary.

- [ ] **Step 4: Implement the enhanced evaluation orchestration behind the preserved entrypoint**

Wire `enhanced_eval.py` through a native evaluation module that preserves the recovered `ENHANCED_RAG` artifact names and uses native retriever, rewrite, rerank, and generation components end-to-end.

### Task 3: Correct docs and guardrails around the preserved enhanced path

**Files:**
- Modify: `docs/superpowers/specs/2026-05-18-llamaindex-stack-migration-design.md`
- Modify: `docs/superpowers/plans/2026-05-18-llamaindex-stack-migration.md`
- Verify: `python-rag/tests/test_llamaindex_native_stack.py`

- [x] **Step 1: Rewrite docs around the corrected feature boundary**

Document that `enhanced_eval.py` remains supported, that only the LangChain implementation is being removed, and that the enhanced native path must preserve the recovered `ENHANCED_RAG` contract.

- [x] **Step 2: Restore an architecture guardrail for the enhanced surface**

Run: `python -m pytest python-rag/tests/test_llamaindex_native_stack.py -q`

Expected: command exits with code 0.

### Task 4: Run focused validation on the restored enhanced slice

**Files:**
- Verify: `python-rag/enhanced_eval.py`
- Verify: `python-rag/app/rag/retriever/hybrid_retriever.py`
- Verify: `python-rag/app/rag/retriever/query_rewrite.py`
- Verify: `python-rag/app/rag/retriever/reranker.py`
- Verify: `python-rag/tests/test_llamaindex_native_stack.py`

- [x] **Step 1: Run the focused guardrail red-green cycle for the enhanced surface**

Run: `python -m pytest python-rag/tests/test_llamaindex_native_stack.py -k enhanced_surface_is_preserved -q`

Expected: first FAIL because the files are missing, then PASS after the files are restored.

- [x] **Step 2: Run narrow syntax and import checks for the restored modules**

Run:

`python -m py_compile python-rag/enhanced_eval.py`

`python -m py_compile python-rag/app/rag/retriever/hybrid_retriever.py`

`python -m py_compile python-rag/app/rag/retriever/query_rewrite.py`

`python -m py_compile python-rag/app/rag/retriever/reranker.py`

`python -c "import pathlib, sys; sys.path.insert(0, str(pathlib.Path('python-rag').resolve())); import enhanced_eval; from app.rag.retriever.hybrid_retriever import HybridRetriever; from app.rag.retriever.query_rewrite import QueryRewritePipeline; from app.rag.retriever.reranker import RerankerPipeline; print('imports ok')"`

Expected: all commands exit with code 0.

