"""Shared query embedding spec for AutoDL formal embedding scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryEmbeddingSpec:
    """One query embedding cache target independent of final formal run execution."""

    cache_id: str
    pipeline: str
    query_text_source: str

