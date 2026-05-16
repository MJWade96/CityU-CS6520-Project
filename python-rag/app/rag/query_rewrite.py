"""
Query Rewrite Module

Implements query rewriting strategies:
1. Rule-based rewriting (medical dictionary, abbreviation expansion)
2. LLM-based rewriting (lightweight)
"""

import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI

from app.rag.eval_shared import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    EvaluationLLMConfig,
    RateLimiter,
    create_async_client,
    get_qwen_completion_kwargs,
    get_qwen_langchain_kwargs,
    parse_optional_bool_env,
)


def _get_env_with_fallback(primary_name: str, fallback_name: str, default: str) -> str:
    """Read a rewrite-specific env var and fall back to the shared LLM env."""
    return os.getenv(primary_name, os.getenv(fallback_name, default))


def _get_query_rewrite_enable_thinking(default: Optional[bool] = False) -> Optional[bool]:
    """Resolve query-rewrite thinking config independently from final generation."""
    shared_default = parse_optional_bool_env("RAG_LLM_ENABLE_THINKING", default=default)
    return parse_optional_bool_env(
        "RAG_QUERY_REWRITE_ENABLE_THINKING",
        default=shared_default,
    )


class MedicalDictionaryRewriter:
    """
    Rule-based query rewriter using medical dictionary

    Features:
    - Synonym replacement
    - Abbreviation expansion
    - Spelling correction
    """

    # Medical abbreviation mappings
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

    # Common synonym mappings
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

    # Chinese medical terms (simplified)
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

    def __init__(self):
        """Initialize the dictionary rewriter"""
        self.abbreviations = self.ABBREVIATIONS
        self.synonyms = self.SYNONYMS
        self.chinese_terms = self.CHINESE_TERMS

    def expand_abbreviations(self, query: str) -> str:
        """Expand medical abbreviations in query"""
        query_lower = query.lower()
        expanded = query_lower

        for abbr, full_form in self.abbreviations.items():
            if abbr in query_lower:
                expanded = expanded.replace(abbr, f"{full_form} ({abbr})")

        return expanded

    def replace_synonyms(self, query: str) -> str:
        """Replace common terms with medical terminology"""
        result = query.lower()

        for common, medical in self.synonyms.items():
            if common in result:
                result = result.replace(common, medical)

        return result

    def expand_chinese_terms(self, query: str) -> str:
        """Expand Chinese medical terms"""
        result = query

        for short, full in self.chinese_terms.items():
            if short in result:
                result = result.replace(short, f"{full} ({short})")

        return result

    def rewrite(self, query: str, strategies: List[str] = None) -> str:
        """
        Apply rewriting strategies.

        Args:
            query: Original query
            strategies: List of strategies to apply
                       ['abbreviations', 'synonyms', 'chinese']

        Returns:
            Rewritten query
        """
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
    """
    LLM-based query rewriter

    Uses lightweight LLM to rewrite queries for better retrieval
    """

    REWRITE_PROMPT = """You are a query rewriting assistant. Your task is to rewrite medical questions to make them more suitable for retrieval.

Guidelines:
- Keep the core meaning
- Expand medical terms
- Remove ambiguity
- Make it more specific and detailed

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
        """
        Initialize LLM rewriter.

        Args:
            provider: LLM provider
            model: Model name/ID
            api_key: API key
            base_url: API base URL
            temperature: Sampling temperature
            max_tokens: Max generation tokens
        """
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

        # Get API credentials
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

        llm_config = EvaluationLLMConfig(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            base_url=self.base_url,
            api_key=self.api_key,
            enable_thinking=self.enable_thinking,
        )
        llm_kwargs = {
            "max_tokens": self.max_tokens,
            "api_key": self.api_key,
            "base_url": self.base_url,
            **get_qwen_langchain_kwargs(llm_config),
        }
        self.completion_kwargs = {
            **get_qwen_completion_kwargs(llm_config),
            "max_tokens": self.max_tokens,
        }
        self.async_client = create_async_client(llm_config)
        self.llm = ChatOpenAI(**llm_kwargs)

    def rewrite(self, query: str) -> str:
        """
        Rewrite query using LLM.

        Args:
            query: Original query

        Returns:
            Rewritten query
        """
        prompt = self.REWRITE_PROMPT.format(query=query)

        try:
            response = self.llm.invoke(prompt)
            rewritten = response.content.strip()
            return rewritten
        except Exception as e:
            # Fallback to original query
            print(f"LLM rewrite failed: {e}")
            return query

    async def arewrite(
        self,
        query: str,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        api_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> str:
        """Rewrite query asynchronously."""
        prompt = self.REWRITE_PROMPT.format(query=query)

        try:
            if api_semaphore:
                async with api_semaphore:
                    if rate_limiter:
                        await rate_limiter.acquire()
                    completion = await self.async_client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        **self.completion_kwargs,
                    )
            else:
                if rate_limiter:
                    await rate_limiter.acquire()
                completion = await self.async_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    **self.completion_kwargs,
                )

            content = (
                completion.choices[0].message.content
                or completion.choices[0].message.reasoning_content
                or ""
            )
            return content.strip() or query
        except Exception as e:
            print(f"LLM rewrite failed: {e}")
            return query


class QueryRewritePipeline:
    """
    Complete query rewrite pipeline

    Combines:
    1. Rule-based rewriting
    2. LLM-based rewriting (optional)
    """

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
        """
        Initialize rewrite pipeline.

        Args:
            use_dict: Use dictionary-based rewriting
            use_llm: Use LLM-based rewriting
            llm_provider: LLM provider
            llm_model: LLM model
            api_key: API key
            base_url: API base URL
        """
        # Initialize components
        if use_dict:
            self.dict_rewriter = MedicalDictionaryRewriter()
        else:
            self.dict_rewriter = None

        if use_llm:
            self.llm_rewriter = LLMQueryRewriter(
                provider=llm_provider,
                model=llm_model,
                api_key=api_key,
                base_url=base_url,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
                enable_thinking=llm_enable_thinking,
            )
        else:
            self.llm_rewriter = None

    def rewrite_with_options(
        self,
        query: str,
        *,
        use_llm: Optional[bool] = None,
    ) -> Tuple[str, List[str]]:
        """Rewrite query with an optional per-call LLM toggle."""
        all_queries = [query]

        if self.dict_rewriter:
            all_queries[0] = self.dict_rewriter.rewrite(query)

        llm_enabled = self.llm_rewriter is not None if use_llm is None else (
            self.llm_rewriter is not None and use_llm
        )
        if llm_enabled:
            all_queries[0] = self.llm_rewriter.rewrite(all_queries[0])

        return all_queries[0], all_queries

    async def arewrite(
        self,
        query: str,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        api_semaphore: Optional[asyncio.Semaphore] = None,
        use_llm: Optional[bool] = None,
    ) -> Tuple[str, List[str]]:
        """Rewrite query asynchronously."""
        all_queries = [query]

        if self.dict_rewriter:
            dict_rewritten = self.dict_rewriter.rewrite(query)
            all_queries[0] = dict_rewritten

        llm_enabled = self.llm_rewriter is not None if use_llm is None else (
            self.llm_rewriter is not None and use_llm
        )
        if llm_enabled:
            llm_rewritten = await self.llm_rewriter.arewrite(
                all_queries[0],
                rate_limiter=rate_limiter,
                api_semaphore=api_semaphore,
            )
            all_queries[0] = llm_rewritten

        return all_queries[0], all_queries
