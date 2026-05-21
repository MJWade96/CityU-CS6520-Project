Background: current RAG implementation uses StatPearls, tests with MedQA-USMLE

Next Steps:
1. **Expand data source beyond StatPearls**  
   - Ingest peer-reviewed journals, systematic reviews, clinical guidelines 
   - integrate diverse document formats 
   - Develop pipelines for PDF extraction & structured conversion  
   - expanded corpus will be evaluated against the StatPearls-only baseline(metrics: answer accuracy, retrieval recall)

2. **Advanced RAG Ablation & Optimization**  
   - *Ablation Part A*: 
    - embedding model, retrieval depth (k), hybrid fusion weight (α), reranker input count, chunk size(metrics: answer accuracy, retrieval recall)
    - parent-child chunking
    - FAISS index (IVF/HNSW) & similarity metrics (cosine, inner, L2)  
    - Query enhancement (one of multi-query generation, query expansion)  

3. **GraphRAG Implementation**  
   - *Phase A*: Entity/relation extraction + graph construction  
   - *Phase B*: Preliminary retrieval + integration with generation pipeline  
   - Evaluate vs. Advanced RAG (single-hop → multi-hop traversal)  

4. **Cross-benchmark Evaluation**  
   - Extend beyond MedQA (e.g., PubMedQA, DDXPlus)  
   - Multi-dimensional assessment: precision/recall, faithfulness, semantic similarity, evidence quality  
   - Systematic error categorization 