# RAG 项目待确认问题与后续计划

## 1. 项目背景

- 本人是计算机科学专业背景，没有医学知识。
- 当前 RAG 实现使用 StatPearls 作为语料。
- 当前评测数据集为 MedQA-USMLE。
- MedQA-USMLE 当前没有 reference context，因此无法直接计算 recall。
- 后续可能继续评测 PubMedQA、MMLU-Med、MedXpertQA。

## 2. 评测框架相关问题

### 2.1 是否适合使用 RAGAS

在当前条件下，是否适合使用 RAGAS：

- 项目使用 MedQA-USMLE 进行评测。
- MedQA-USMLE 没有 reference context，无法计算 recall。
- 本人没有医学知识，难以手动设计医学评测标准。

RAGAS 官方文档：

<https://docs.ragas.io/en/stable/getstarted/>

### 2.2 RAGAS 是否能减少手动评测代码

如果后续评测 PubMedQA、MMLU-Med、MedXpertQA，是否可以使用 RAGAS 框架，避免手动编写评测代码？

## 3. 实验顺序问题

当前需要确认优先级：

- 先扩展语料库；
- 还是先做消融实验。

## 4. 项目架构草稿

在使用 LlamaIndex、GraphRAG，以及可能使用 RAGAS 的情况下，以下架构是否需要调整：

```text
project/
├── schemas/
├── data/
│   ├── raw/                 # 原始文档，例如 pdf/html/json/xml
│   ├── processed/           # 清洗与结构化结果
│   ├── ingestion/           # 多源采集与格式统一
│   └── parsing/             # PDF、HTML、指南结构化抽取
├── corpus/v{n}/
├── evaluation/              # 消费 runs/，不关心后端
│   ├── benchmarks/          # 评测数据集
│   ├── metrics/
│   ├── error_taxonomy/
│   └── compare/             # Advanced RAG 与 GraphRAG 对比
├── backends/
│   ├── advanced_rag/
│   │   └── query_enhance/
│   └── graphrag/
├── runs/{backend}/{run_id}/ # 两端共同落地区，仍按 RunRecord 契约
├── experiments/             # 编排器，可同时驱动两端做消融实验
└── scripts/
```

由于语料库只计划使用 StatPearls 和 TextBook，`data/` 目录结构是否也需要相应调整？

## 5. 后续任务计划

### 5.1 2026.05.27 至 2026.06.09

#### 5.1.1 扩展数据源

- 在 StatPearls 之外扩展 TextBook 语料。
- TextBook 来源为 MedScore：<https://github.com/Heyuan9/MedScore>。
- 当前 StatPearls 也来自 MedScore。
- 尽量复用 MedScore 中的代码。
- 扩展后的语料库需要与仅使用 StatPearls 的 baseline 进行对比评测。

#### 5.1.2 Advanced RAG 消融实验

需要评测的参数及取值：

- Embedding model：
  - MedCPT
  - BGE-m3（已构建）
  - bge-large-en-v1.5
- Retrieval depth `k`：
  - 3
  - 5
  - 10
- `alpha`：
  - 0.0
  - 0.25
  - 0.5
  - 0.75
  - 1.0
- Reranker input count：
  - 根据 `k` 决定，设置为 `k` 的 2、4、8 倍。

实现方式：

- Embedding model 和 reranker 通过 API 调用。
- 参考 GitHub 仓库中的代码部署 MedCPT，例如 MedScore。
- 或者使用 HuggingFace Inference 服务。

### 5.2 2026.06.09 至 2026.07.21

#### 5.2.1 可能进行的改造

以下任务可能做，也可能不做；如果要做，需要确认代码改造是否复杂：

- 从 FAISS 切换为 Caliby：<https://github.com/zxjcarrot/caliby>
- Query enhancement，例如 multi-query generation 或 query expansion。
- FAISS index 消融实验，包括 Flat、IVF、HNSW。

#### 5.2.2 GraphRAG 实现

- 实现 GraphRAG。
- 对比 GraphRAG 与 Advanced RAG。

#### 5.2.3 跨数据集评测

- 将评测范围从 MedQA 扩展到 PubMedQA、MMLU-Med、MedXpertQA。
- 评估是否使用 RAGAS。
- 做系统性错误分类。
- 错误分析方式参考 `qualitative error analysis.md`。
- 报告中准备展示用于分析的题目本身以及模型的推理过程。

## 6. 结果缓存问题

建议缓存以下结果：

- chunk embeddings
- FAISS index
- retrieve top-k
- rerank outputs
- final prompts
- LLM outputs

推荐流程：

```text
Embedding Ablation
    ↓
保存 retrieval outputs

Reranker Ablation
    ↓
保存 rerank outputs

Generator Ablation
    ↓
最终 QA evaluation
```

可使用的缓存方式：

- LlamaIndex 默认支持：
  - persist storage
  - persist vector store
  - node cache
- 使用 `index.storage_context.persist("./storage")` 与 `load_index_from_storage(...)`。
- 将 retrieve 结果持久化保存。

Retrieve 结果保存示例：

```json
{
  "question_id": "123",
  "query": "...",
  "topk_chunks": [
    {
      "chunk_id": "...",
      "score": 0.83
    }
  ]
}
```
