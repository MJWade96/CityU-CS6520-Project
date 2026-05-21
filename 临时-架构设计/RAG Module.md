# RAG 系统模块架构
## Indexing（索引）
### Chunk Optimization（块优化）
+ Small-to-Big  
+ Sliding Window  
+ Summary  
+ Metadata Attachment  

### Structural Organization（结构化组织）
+ Hierarchical Index  
+ KG Organization  

---

## Pre-Retrieval（检索前处理）
### Query Routing（查询路由）
+ Metadata Filter  
+ Metadata Router/Filter  

### Query Expansion（查询扩展）
+ CoVe  
+ Multi Query  
+ SubQuery  

### Query Transformation（查询变换）
+ Rewrite  
+ Step-back Prompting  
+ (Reverse) HyDE  

### Query Construction（查询构建）
+ Text-to-cypher  
+ Text-to-SQL  

---

## Retrieval（检索）
### Retriever Selection（检索器选择）
+ Sparse Retriever  
+ Dense Retriever  
+ Mix Retriever  

### Retriever Tuning（检索器调优）
+ SFT  
+ LSR  
+ RL  
+ Adapter  

---

## Post-Retrieval（检索后处理）
### Rerank（重排序）
+ Rule-Based  
+ Model-Based  
+ LLM-Based  

### Compression/Selection（压缩/筛选）
+ (Long)LLMLingua  
+ Recomp  
+ Tagging-Filter  
+ Selective Context  
+ LLM-Critique  

---

## Generation（生成）
### Generator（生成器）
+ Cloud API-base  
+ On-premises  

### Generator Tuning（生成器调优）
+ SFT  
+ RL  
+ Dual FT  

---

## Orchestration（编排）
### Scheduling（调度）
+ Rule-base  
+ Prompt-base  
+ Tuning-base  

### Fusion（融合）
+ Possibility Ensemble  
+ RRF  
