# LlamaIndex Stack Migration Design

## Goal

Replace the mixed retrieval stack with a single native LlamaIndex implementation behind the original script and module names.

## Scope

- Keep the current MedQA result schema, dataset splits, top-k search behavior, and answer extraction flow.
- Keep the original primary entrypoint names: `build_vector_index.py`, `complete_eval.py`, `sample_validation.py`, and `enhanced_eval.py`.
- Implement retrieval and answering with `HuggingFaceEmbedding`, `OpenAILike`, `FaissVectorStore`, and `VectorStoreIndex` query capability.
- Preserve the enhanced-evaluation feature surface while replacing its LangChain-bound implementation modules with native LlamaIndex abstractions.
- Remove parallel `llamaindex_*` entrypoints and the deprecated LangChain-only dependencies.

## Non-Goals

- No claim of better answer quality.
- No GraphRAG or broader orchestration redesign.
- No retention of the deprecated LangChain implementation behind the enhanced path.
- No dual-stack fallback or script-level compatibility layer that keeps LangChain alive at runtime.

## Design

### Canonical module boundary

The canonical runtime modules are now the old module names: `app/rag/retriever/vector_store.py` and `app/rag/evaluation/naive_rag_eval.py`. They are implemented natively with LlamaIndex APIs instead of wrapping or adapting a second stack.

### Index construction

Use the same combined corpus source, but convert it directly into LlamaIndex `Document` objects. Index persistence follows the documented FAISS pattern:

- `HuggingFaceEmbedding` for embeddings
- `FaissVectorStore` inside `StorageContext`
- `VectorStoreIndex.from_documents(...)` for build
- `index.storage_context.persist(...)` and `FaissVectorStore.from_persist_dir(...)` for persistence

The repo keeps a single persisted store location at `<data-root>/vector_store/faiss_index`. After this migration that directory must be rebuilt with the native builder because its on-disk format changes.

### Evaluation path

`complete_eval.py` remains the primary full-evaluation entrypoint, but it now routes to the native `naive_rag_eval.py` implementation. The evaluation path uses `OpenAILike` plus `VectorStoreIndex.as_query_engine(...)` for answer generation, while keeping shared progress artifacts and answer extraction unchanged.

`sample_validation.py` remains a supported root entrypoint, but its implementation should live under `app/rag/evaluation/` so the root script stays a thin CLI wrapper rather than carrying evaluation logic directly.

`enhanced_eval.py` remains a supported feature entrypoint rather than a disposable LangChain-only script. Its recovered historical contract includes artifact prefix `enhanced_rag_eval`, run name and evaluation type `ENHANCED_RAG`, and checkpoint script names `enhanced_eval_dev` and `enhanced_eval_test`. The restored native implementation must preserve those user-visible names while replacing the underlying retrieval, rewrite, and rerank logic.

### Enhanced retrieval path

The enhanced path keeps the same high-level stages as the deleted implementation: query rewrite, retrieval fusion, reranking, and answer generation. The native mapping is:

- Dense retrieval from `VectorStoreIndex.as_retriever(...)`
- Sparse retrieval from `BM25Retriever`
- Fusion from `QueryFusionRetriever`
- Deterministic medical term expansion plus native query transforms for rewrite
- `SentenceTransformerRerank` or a custom native node postprocessor for reranking

The historical implementation combined dense retrieval, BM25 retrieval, Reciprocal Rank Fusion, dictionary and LLM query rewrite, and cross-encoder reranking. The migration must preserve that feature boundary even if some concrete native components change during implementation.

### Runtime prerequisites

Document prerequisites in terms of the resolved data root instead of hard-coded `python-rag/data/...` paths. In this repo the scripts use `python-rag/data/` if present; otherwise they fall back to the sibling `RAG_Medical_Data/` directory unless `RAG_DATA_DIR` overrides it. Resume guidance should only mention the scripts that actually persist and consume checkpoints: `complete_eval.py`, `sample_validation.py`, and `evaluate_no_rag.py`.

### Expected behavioral differences

The migration is designed to minimize quality drift, but the native query engine will not be prompt-identical to the removed stack. Minor score/order/wording differences are expected because retrieval and response synthesis are now delegated to native LlamaIndex abstractions.

## Validation

- Syntax-check the new/changed Python files.
- Run a narrow smoke import for the retained primary entrypoints, including `enhanced_eval.py`.
- Run a focused architecture guardrail test that fails if runtime Python files reintroduce `langchain` imports, if the removed `llamaindex_*` entrypoints return, or if the enhanced surface disappears again.
- If corpus/index assets exist later, rebuild `faiss_index` and run the sample/full evaluation flows against the same dataset slices as before.
