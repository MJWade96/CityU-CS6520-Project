"""
FastAPI Application for Medical RAG System
Provides REST API endpoints for the RAG pipeline
"""

import os
import sys
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import (
    create_rag_pipeline,
    RAGConfig,
    RAGResult,
    MedicalRAGPipeline
)


# Global pipeline instance
pipeline: Optional[MedicalRAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global pipeline
    
    # Startup: Initialize the RAG pipeline
    print("Initializing Medical RAG Pipeline...")
    pipeline = create_rag_pipeline()
    print(f"Pipeline initialized: {pipeline.get_stats()}")
    
    yield
    
    # Shutdown: Cleanup
    print("Shutting down Medical RAG Pipeline...")


# Create FastAPI application
app = FastAPI(
    title="Medical RAG API",
    description="Retrieval-Augmented Generation API for Medical Diagnosis Support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for RAG query"""
    query: str = Field(..., description="Medical question to answer", min_length=1)
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    config: Optional[Dict[str, Any]] = Field(None, description="Optional RAG configuration")


class Source(BaseModel):
    """Source document model"""
    title: str
    source: str
    category: str
    relevance_score: float
    chunk_index: Optional[int] = None


class QueryResponse(BaseModel):
    """Response model for RAG query"""
    success: bool
    query: str
    answer: str
    sources: List[Source]
    confidence: float
    processing_time: float
    retrieval_time: float


class SystemStatus(BaseModel):
    """System status model"""
    status: str
    is_initialized: bool
    document_count: int
    config: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    stats = pipeline.get_stats()
    
    return SystemStatus(
        status="operational",
        is_initialized=stats.get('is_initialized', False),
        document_count=stats.get('vector_store_stats', {}).get('document_count', 0),
        config=stats.get('config', {})
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a medical query through the RAG pipeline.
    
    This endpoint accepts a medical question and returns an evidence-based
    answer with source citations from the medical knowledge base.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Process the query
        result: RAGResult = pipeline.query(request.query)
        
        # Build response
        return QueryResponse(
            success=True,
            query=result.query,
            answer=result.answer,
            sources=[
                Source(**source) for source in result.sources
            ],
            confidence=result.confidence,
            processing_time=result.total_time,
            retrieval_time=result.retrieval_result.retrieval_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/search")
async def search_documents(
    q: str = Query(..., description="Search query", min_length=1),
    k: int = Query(5, description="Number of results", ge=1, le=20)
):
    """
    Search for relevant documents without generating an answer.
    
    This endpoint performs retrieval only and returns the matched documents.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        retrieval_result = pipeline.retrieve(q, k=k)
        
        return {
            "success": True,
            "query": q,
            "documents": [
                {
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in zip(retrieval_result.documents, retrieval_result.scores)
            ],
            "retrieval_time": retrieval_result.retrieval_time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/reload")
async def reload_knowledge_base():
    """
    Reload the knowledge base.
    
    This endpoint reinitializes the RAG pipeline with the knowledge base.
    """
    global pipeline
    
    try:
        pipeline = create_rag_pipeline()
        stats = pipeline.get_stats()
        
        return {
            "success": True,
            "message": "Knowledge base reloaded successfully",
            "stats": stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


# Run the application
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
