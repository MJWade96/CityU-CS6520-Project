Architure
  1) 顶层架构（分层 + 可替换）

  

  2) 核心设计原则（为后续 ablation 做准备）

  1. 统一接口，后端可插拔
    - Embedder, Retriever, Reranker, Generator, GraphStore 全部定义抽象接口。
    - 每个实验只替换实现，不改主流程。
  2. 配置驱动实验
    - 一个 experiment_config 控制：embedding model、k、α、chunk size、faiss index type、similarity metric、rerank_topn、query enhance strategy。
    - 避免把实验组合写死在代码里。
  3. 评测与主流程解耦
    - 推理流程只返回结构化结果：answer, retrieved_docs, citations, trace。
    - evaluation/ 独立消费结果，便于跨基准复用（MedQA / PubMedQA / DDXPlus）。
  4. GraphRAG 与 Advanced RAG 并行可比较
    - 统一 retrieve(query) 协议：
        - Advanced RAG 返回文档证据；
      - GraphRAG 返回子图证据 + 映射到文本证据。

    - 评测层不关心后端细节。


  4) 建议的数据契约（最关键）

  - Document: doc_id, source, title, publish_date, sections[]
  - Chunk: chunk_id, doc_id, parent_id?, text, tokens, metadata
  - RetrievalHit: id, score, source_type(vector|bm25|graph), evidence
  - RunTrace: query_variant, retriever_params, rerank_params, latency_breakdown
  - EvalRecord: question_id, benchmark, answer, gold, metrics, error_tags

  这样后面做 recall/faithfulness/evidence quality 时不用反复改字段。

  5) 执行顺序（最小风险）

  1. 先搭 Document/Chunk/RetrievalHit/EvalRecord schema
  2. 再做 ingestion+parsing（先 PDF）
  3. 接着接入 baseline advanced RAG（StatPearls + MedQA 保持可复现）
  4. 然后加 ablation orchestrator
  5. 最后并行接 GraphRAG Phase A/B 与 cross-benchmark