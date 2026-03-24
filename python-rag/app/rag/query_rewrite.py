"""
Query Rewrite Module

Implements query rewriting strategies:
1. Rule-based rewriting (medical dictionary, abbreviation expansion)
2. LLM-based rewriting (lightweight)
3. Query expansion (Multi-Query strategy)
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI


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
        provider: str = "Qwen3-4B",
        model: str = "8606056bfe0c49448d92587452d1f2fc",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 200,
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
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API credentials
        self.api_key = api_key or os.getenv(
            "DEEPSEEK_API_KEY", "6fcecb364d0647d2883e7f1d3f19d5b9"
        )
        self.base_url = base_url or "https://wishub-x6.ctyun.cn/v1"

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
        )

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


class QueryExpander:
    """
    Query expansion using Multi-Query strategy

    Generates multiple semantically related queries to improve recall
    """

    EXPANSION_PROMPT = """Based on the original question, generate 3-5 semantically related but diverse questions that cover different aspects.

Guidelines:
- Keep diversity
- Cover different angles
- Maintain medical accuracy

Original question: {query}

Generated questions (3-5):"""

    def __init__(
        self,
        provider: str = "Qwen3-4B",
        model: str = "8606056bfe0c49448d92587452d1f2fc",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        num_expansions: int = 3,
    ):
        """
        Initialize query expander.

        Args:
            provider: LLM provider
            model: Model name/ID
            api_key: API key
            base_url: API base URL
            num_expansions: Number of expanded queries to generate
        """
        self.provider = provider
        self.model = model
        self.num_expansions = num_expansions

        # Get API credentials
        self.api_key = api_key or os.getenv(
            "DEEPSEEK_API_KEY", "6fcecb364d0647d2883e7f1d3f19d5b9"
        )
        self.base_url = base_url or "https://wishub-x6.ctyun.cn/v1"

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0.3,
            max_tokens=300,
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def expand(self, query: str) -> List[str]:
        """
        Expand query into multiple related queries.

        Args:
            query: Original query

        Returns:
            List of expanded queries (including original)
        """
        prompt = self.EXPANSION_PROMPT.format(query=query)

        try:
            response = self.llm.invoke(prompt)
            expanded_queries = response.content.strip().split("\n")

            # Clean up queries
            cleaned = []
            for q in expanded_queries:
                q = q.strip()
                # Remove numbering
                if q and q[0].isdigit():
                    q = q[2:].strip()
                if q and len(q) > 5:
                    cleaned.append(q)

            # Add original query
            if query not in cleaned:
                cleaned.insert(0, query)

            return cleaned[: self.num_expansions + 1]

        except Exception as e:
            print(f"Query expansion failed: {e}")
            return [query]


class QueryRewritePipeline:
    """
    Complete query rewrite pipeline

    Combines:
    1. Rule-based rewriting
    2. LLM-based rewriting (optional)
    3. Query expansion (optional)
    """

    def __init__(
        self,
        use_dict: bool = True,
        use_llm: bool = True,
        use_expansion: bool = False,
        llm_provider: str = "deepseek",
        llm_model: str = "2656053fa69c4c2d89c5a691d9d737c3",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize rewrite pipeline.

        Args:
            use_dict: Use dictionary-based rewriting
            use_llm: Use LLM-based rewriting
            use_expansion: Use query expansion
            llm_provider: LLM provider
            llm_model: LLM model
            api_key: API key
            base_url: API base URL
        """
        self.use_dict = use_dict
        self.use_llm = use_llm
        self.use_expansion = use_expansion

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
            )
        else:
            self.llm_rewriter = None

        if use_expansion:
            self.expander = QueryExpander(
                provider=llm_provider,
                model=llm_model,
                api_key=api_key,
                base_url=base_url,
            )
        else:
            self.expander = None

    def rewrite(self, query: str, mode: str = "single") -> Tuple[str, List[str]]:
        """
        Rewrite query.

        Args:
            query: Original query
            mode: 'single' (one query) or 'expanded' (multiple queries)

        Returns:
            Tuple of (primary_rewritten_query, all_queries_for_retrieval)
        """
        all_queries = [query]

        # Step 1: Dictionary-based rewriting
        if self.dict_rewriter:
            dict_rewritten = self.dict_rewriter.rewrite(query)
            all_queries[0] = dict_rewritten

        # Step 2: LLM rewriting
        if self.llm_rewriter and mode == "single":
            llm_rewritten = self.llm_rewriter.rewrite(all_queries[0])
            all_queries[0] = llm_rewritten

        # Step 3: Query expansion
        if self.expander and mode == "expanded":
            expanded = self.expander.expand(all_queries[0])
            all_queries = expanded

        return all_queries[0], all_queries

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "use_dict_rewriting": self.dict_rewriter is not None,
            "use_llm_rewriting": self.llm_rewriter is not None,
            "use_query_expansion": self.expander is not None,
        }


if __name__ == "__main__":
    # Test query rewriting
    print("Testing Query Rewrite Module...")

    # Test dictionary rewriter
    print("\n=== Dictionary-based Rewriting ===")
    dict_rewriter = MedicalDictionaryRewriter()

    test_queries = [
        "What is MI?",
        "Treatment for high blood pressure",
        "心梗的症状是什么",
        "How to treat diabetes?",
    ]

    for query in test_queries:
        rewritten = dict_rewriter.rewrite(query)
        print(f"\nOriginal:  {query}")
        print(f"Rewritten: {rewritten}")

    # Test LLM rewriter
    print("\n=== LLM-based Rewriting ===")
    llm_rewriter = LLMQueryRewriter()

    test_query = "What are the symptoms of heart attack?"
    rewritten = llm_rewriter.rewrite(test_query)
    print(f"\nOriginal:  {test_query}")
    print(f"Rewritten: {rewritten}")

    # Test query expansion
    print("\n=== Query Expansion ===")
    expander = QueryExpander()

    expanded = expander.expand(test_query)
    print(f"\nOriginal:  {test_query}")
    print(f"Expanded queries:")
    for i, q in enumerate(expanded, 1):
        print(f"  {i}. {q}")

    # Test pipeline
    print("\n=== Complete Pipeline ===")
    pipeline = QueryRewritePipeline(
        use_dict=True,
        use_llm=True,
        use_expansion=False,
    )

    primary, all_queries = pipeline.rewrite("What is MI?", mode="single")
    print(f"\nOriginal:  What is MI?")
    print(f"Primary:   {primary}")
    print(f"All queries: {all_queries}")
