"""Native query-rewrite helpers for the enhanced RAG path."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core import QueryBundle
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform
from llama_index.llms.openai_like import OpenAILike

from ..evaluation.eval_shared import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    EvaluationLLMConfig,
    RateLimiter,
    get_qwen_openai_like_kwargs,
    parse_optional_bool_env,
)


def _get_env_with_fallback(primary_name: str, fallback_name: str, default: str) -> str:
    return os.getenv(primary_name, os.getenv(fallback_name, default))


def _get_query_rewrite_enable_thinking(default: Optional[bool] = False) -> Optional[bool]:
    shared_default = parse_optional_bool_env("RAG_LLM_ENABLE_THINKING", default=default)
    return parse_optional_bool_env(
        "RAG_QUERY_REWRITE_ENABLE_THINKING",
        default=shared_default,
    )


class MedicalDictionaryRewriter:
    """Deterministic rule layer for abbreviation and terminology expansion."""

    ABBREVIATIONS = {
        "mi": "myocardial infarction",
        "cad": "coronary artery disease",
        "hf": "heart failure",
        "afib": "atrial fibrillation",
        "copd": "chronic obstructive pulmonary disease",
        "dm": "diabetes mellitus",
        "htn": "hypertension",
        "ckd": "chronic kidney disease",
        "stroke": "cerebrovascular accident",
        "pe": "pulmonary embolism",
        "dvt": "deep vein thrombosis",
        "gi": "gastrointestinal",
        "gu": "genitourinary",
        "cns": "central nervous system",
        "ans": "autonomic nervous system",
    }

    SYNONYMS = {
        "heart attack": "myocardial infarction",
        "high blood pressure": "hypertension",
        "low blood pressure": "hypotension",
        "high blood sugar": "hyperglycemia",
        "low blood sugar": "hypoglycemia",
        "kidney failure": "renal failure",
        "liver failure": "hepatic failure",
        "lung infection": "pneumonia",
        "blood cancer": "leukemia",
        "bone cancer": "osteosarcoma",
        "skin cancer": "melanoma",
        "chest pain": "angina",
        "shortness of breath": "dyspnea",
        "difficulty breathing": "dyspnea",
        "headache": "cephalgia",
        "dizziness": "vertigo",
        "fatigue": "tiredness",
        "nausea": "feeling sick",
        "vomiting": "emesis",
        "diarrhea": "loose stools",
        "constipation": "difficulty passing stools",
    }

    CHINESE_TERMS = {
        "心梗": "心肌梗死",
        "冠心病": "冠状动脉疾病",
        "心衰": "心力衰竭",
        "房颤": "心房颤动",
        "慢阻肺": "慢性阻塞性肺疾病",
        "糖尿病": "糖尿病",
        "高血压": "高血压病",
        "肾病": "肾脏疾病",
        "中风": "脑卒中",
        "肺炎": "肺部感染",
    }

    def expand_abbreviations(self, query: str) -> str:
        rewritten = query.lower()
        for abbr, full_form in self.ABBREVIATIONS.items():
            if abbr in rewritten:
                rewritten = rewritten.replace(abbr, f"{full_form} ({abbr})")
        return rewritten

    def replace_synonyms(self, query: str) -> str:
        rewritten = query.lower()
        for common, medical in self.SYNONYMS.items():
            if common in rewritten:
                rewritten = rewritten.replace(common, medical)
        return rewritten

    def expand_chinese_terms(self, query: str) -> str:
        rewritten = query
        for short, full in self.CHINESE_TERMS.items():
            if short in rewritten:
                rewritten = rewritten.replace(short, f"{full} ({short})")
        return rewritten

    def rewrite(self, query: str, strategies: Optional[List[str]] = None) -> str:
        if strategies is None:
            strategies = ["abbreviations", "synonyms", "chinese"]

        rewritten = query
        if "abbreviations" in strategies:
            rewritten = self.expand_abbreviations(rewritten)
        if "synonyms" in strategies:
            rewritten = self.replace_synonyms(rewritten)
        if "chinese" in strategies:
            rewritten = self.expand_chinese_terms(rewritten)
        return rewritten


class LLMQueryRewriter:
    """Native OpenAILike-backed query rewriter."""

    REWRITE_PROMPT = """You are a query rewriting assistant. Rewrite the medical query for retrieval while keeping the original meaning.

Original question: {query}

Rewritten question:"""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ):
        self.provider = provider or _get_env_with_fallback(
            "RAG_QUERY_REWRITE_PROVIDER",
            "RAG_LLM_PROVIDER",
            DEFAULT_PROVIDER,
        )
        self.model = model or _get_env_with_fallback(
            "RAG_QUERY_REWRITE_MODEL",
            "RAG_LLM_MODEL",
            DEFAULT_MODEL,
        )
        self.temperature = (
            temperature
            if temperature is not None
            else float(
                _get_env_with_fallback(
                    "RAG_QUERY_REWRITE_TEMPERATURE",
                    "RAG_LLM_TEMPERATURE",
                    "0.1",
                )
            )
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.getenv("RAG_QUERY_REWRITE_MAX_TOKENS", "200"))
        )
        self.enable_thinking = (
            enable_thinking
            if enable_thinking is not None
            else _get_query_rewrite_enable_thinking(default=False)
        )
        self.api_key = api_key or _get_env_with_fallback(
            "RAG_QUERY_REWRITE_API_KEY",
            "RAG_LLM_API_KEY",
            DEFAULT_API_KEY,
        )
        self.base_url = base_url or _get_env_with_fallback(
            "RAG_QUERY_REWRITE_BASE_URL",
            "RAG_LLM_BASE_URL",
            DEFAULT_BASE_URL,
        )
        self.llm = OpenAILike(
            **get_qwen_openai_like_kwargs(
                EvaluationLLMConfig(
                    provider=self.provider,
                    model=self.model,
                    temperature=self.temperature,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    enable_thinking=self.enable_thinking,
                )
            )
        )

    def rewrite(self, query: str) -> str:
        response = self.llm.complete(
            self.REWRITE_PROMPT.format(query=query),
            max_tokens=self.max_tokens,
        )
        rewritten = (response.text or "").strip()
        return rewritten or query

    async def arewrite(
        self,
        query: str,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        api_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> str:
        prompt = self.REWRITE_PROMPT.format(query=query)
        if api_semaphore:
            async with api_semaphore:
                if rate_limiter:
                    await rate_limiter.acquire()
                response = await self.llm.acomplete(prompt, max_tokens=self.max_tokens)
        else:
            if rate_limiter:
                await rate_limiter.acquire()
            response = await self.llm.acomplete(prompt, max_tokens=self.max_tokens)

        rewritten = (response.text or "").strip()
        return rewritten or query


class MedicalRewriteTransform(BaseQueryTransform):
    """Native query transform that applies deterministic and optional LLM rewrite."""

    def __init__(
        self,
        dictionary_rewriter: Optional[MedicalDictionaryRewriter] = None,
        llm_rewriter: Optional[LLMQueryRewriter] = None,
        *,
        use_llm: bool = False,
    ):
        self.dictionary_rewriter = dictionary_rewriter
        self.llm_rewriter = llm_rewriter
        self.use_llm = use_llm

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        rewritten = query_bundle.query_str
        if self.dictionary_rewriter is not None:
            rewritten = self.dictionary_rewriter.rewrite(rewritten)

        use_llm = metadata.get("use_llm", self.use_llm)
        if use_llm and self.llm_rewriter is not None:
            rewritten = self.llm_rewriter.rewrite(rewritten)

        return QueryBundle(query_str=rewritten)


class QueryRewritePipeline:
    """Query rewrite pipeline that preserves the old surface on native abstractions."""

    def __init__(
        self,
        use_dict: bool = True,
        use_llm: bool = True,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_max_tokens: Optional[int] = None,
        llm_enable_thinking: Optional[bool] = None,
    ):
        self.dict_rewriter = MedicalDictionaryRewriter() if use_dict else None
        self.llm_rewriter = (
            LLMQueryRewriter(
                provider=llm_provider,
                model=llm_model,
                api_key=api_key,
                base_url=base_url,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
                enable_thinking=llm_enable_thinking,
            )
            if use_llm
            else None
        )

    def rewrite_with_options(
        self,
        query: str,
        *,
        use_llm: Optional[bool] = None,
    ) -> Tuple[str, List[str]]:
        rewritten = query
        if self.dict_rewriter is not None:
            rewritten = self.dict_rewriter.rewrite(rewritten)

        llm_enabled = self.llm_rewriter is not None if use_llm is None else (
            self.llm_rewriter is not None and use_llm
        )
        if llm_enabled and self.llm_rewriter is not None:
            rewritten = self.llm_rewriter.rewrite(rewritten)

        return rewritten, [rewritten]

    async def arewrite(
        self,
        query: str,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        api_semaphore: Optional[asyncio.Semaphore] = None,
        use_llm: Optional[bool] = None,
    ) -> Tuple[str, List[str]]:
        rewritten = query
        if self.dict_rewriter is not None:
            rewritten = self.dict_rewriter.rewrite(rewritten)

        llm_enabled = self.llm_rewriter is not None if use_llm is None else (
            self.llm_rewriter is not None and use_llm
        )
        if llm_enabled and self.llm_rewriter is not None:
            rewritten = await self.llm_rewriter.arewrite(
                rewritten,
                rate_limiter=rate_limiter,
                api_semaphore=api_semaphore,
            )

        return rewritten, [rewritten]

    def as_transform(self, *, use_llm: bool = False) -> MedicalRewriteTransform:
        return MedicalRewriteTransform(
            dictionary_rewriter=self.dict_rewriter,
            llm_rewriter=self.llm_rewriter,
            use_llm=use_llm,
        )