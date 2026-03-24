"""
FastAPI Application for Medical RAG System
Provides REST API endpoints for the RAG pipeline
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import create_rag_pipeline, RAGConfig, MedicalRAGSystem, RuleBasedEvaluator

CUSTOM_CONFIG = {
    "base_url": "https://wishub-x6.ctyun.cn/v1",
    "api_key": "6fcecb364d0647d2883e7f1d3f19d5b9",
    "model": "8606056bfe0c49448d92587452d1f2fc",
}


def create_custom_rag_pipeline():
    from app.rag.api_medical_rag import MedicalRAGConfig

    config = MedicalRAGConfig(
        llm_provider="Qwen3-4B",
        llm_model=CUSTOM_CONFIG["model"],
        llm_api_key=CUSTOM_CONFIG["api_key"],
        llm_base_url=CUSTOM_CONFIG["base_url"],
    )
    system = MedicalRAGSystem(config)
    system.initialize()
    return system


pipeline: Optional[MedicalRAGSystem] = None
evaluator = RuleBasedEvaluator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    print("Initializing Medical RAG Pipeline...")
    try:
        pipeline = create_custom_rag_pipeline()
        print(
            f"Pipeline initialized: {len(pipeline.documents) if hasattr(pipeline, 'documents') else 0} documents"
        )
    except Exception as e:
        print(f"Warning: Could not initialize pipeline: {e}")
        pipeline = None

    yield

    print("Shutting down Medical RAG Pipeline...")


app = FastAPI(
    title="Medical RAG API",
    description="Retrieval-Augmented Generation API for Medical Diagnosis Support",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Medical question to answer", min_length=1)
    top_k: Optional[int] = Field(
        5, description="Number of documents to retrieve", ge=1, le=20
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="Optional RAG configuration"
    )


class Source(BaseModel):
    title: str
    source: str
    category: Optional[str] = None
    relevance_score: Optional[float] = None
    chunk_index: Optional[int] = None


class QueryResponse(BaseModel):
    success: bool
    query: str
    answer: str
    sources: List[Source]
    confidence: float
    processing_time: float
    retrieval_time: float


class SystemStatus(BaseModel):
    status: str
    is_initialized: bool
    document_count: int
    config: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/status", response_model=SystemStatus)
async def get_status():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return SystemStatus(
        status="operational",
        is_initialized=pipeline.is_initialized,
        document_count=len(pipeline.documents),
        config={},
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        start_time = time.time()

        documents = pipeline.retrieve(request.query, k=request.top_k or 5)
        retrieval_time = time.time() - start_time

        answer = pipeline.generate(request.query, documents)

        contexts = [doc.page_content for doc in documents]
        eval_result = evaluator.comprehensive_evaluation(
            question=request.query,
            answer=answer,
            contexts=contexts,
            retrieved_docs=documents,
        )

        confidence = eval_result.get("overall_score", 0.5) if eval_result else 0.5
        total_time = time.time() - start_time

        sources = []
        for i, doc in enumerate(documents):
            sources.append(
                Source(
                    title=doc.metadata.get("title", "Unknown"),
                    source=doc.metadata.get("source", "Unknown"),
                    category=doc.metadata.get("category"),
                    relevance_score=1.0 - (i * 0.1),
                    chunk_index=doc.metadata.get("chunk_index"),
                )
            )

        return QueryResponse(
            success=True,
            query=request.query,
            answer=answer,
            sources=sources,
            confidence=confidence,
            processing_time=total_time,
            retrieval_time=retrieval_time,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )


@app.get("/search")
async def search_documents(
    q: str = Query(..., description="Search query", min_length=1),
    k: int = Query(5, description="Number of results", ge=1, le=20),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        start_time = time.time()
        documents = pipeline.retrieve(q, k=k)
        retrieval_time = time.time() - start_time

        return {
            "success": True,
            "query": q,
            "documents": [
                {
                    "content": (
                        doc.page_content[:500] + "..."
                        if len(doc.page_content) > 500
                        else doc.page_content
                    ),
                    "metadata": doc.metadata,
                }
                for doc in documents
            ],
            "retrieval_time": retrieval_time,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/reload")
async def reload_knowledge_base():
    global pipeline

    try:
        pipeline = create_custom_rag_pipeline()

        return {
            "success": True,
            "message": "Knowledge base reloaded successfully",
            "document_count": len(pipeline.documents),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
