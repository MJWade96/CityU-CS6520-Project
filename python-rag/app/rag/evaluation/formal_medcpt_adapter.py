"""Backward-compatible MedCPT name for the generic formal local embedding adapter."""

from __future__ import annotations

from app.rag.evaluation.formal_local_embedding_adapter import LocalEmbeddingFormalRetriever


MedCPTFormalRetriever = LocalEmbeddingFormalRetriever
