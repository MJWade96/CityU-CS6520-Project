"""
Metadata Enhancement Module

Implements metadata generation and enrichment for chunks:
1. LLM-based summary generation
2. Keyword extraction
3. Question generation
4. Entity extraction
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI

from app.rag.eval_shared import build_extra_body, parse_optional_bool_env


class MetadataGenerator:
    """
    Generates metadata for document chunks using LLM
    
    Generates:
    - Summary (50-100 tokens)
    - Keywords (3-5 keywords)
    - Questions (3-5 questions the chunk can answer)
    - Entities (medical entities mentioned)
    """
    
    SUMMARY_PROMPT = """Generate a concise summary of the following medical text.

Guidelines:
- Keep it under 100 words
- Focus on key medical facts
- Include important numbers and dosages if present

Text: {text}

Summary:"""
    
    KEYWORDS_PROMPT = """Extract 3-5 important keywords or key phrases from the following medical text.

Guidelines:
- Focus on medical terms
- Include diseases, symptoms, treatments, and procedures
- Return as a comma-separated list

Text: {text}

Keywords:"""
    
    QUESTIONS_PROMPT = """Generate 3-5 questions that this medical text could answer.

Guidelines:
- Questions should be clinically relevant
- Cover different aspects of the text
- Format as actual questions

Text: {text}

Questions:"""
    
    ENTITIES_PROMPT = """Extract medical entities from the following text.

Categories to extract:
- Diseases and conditions
- Symptoms
- Treatments and medications
- Procedures
- Anatomical terms

Text: {text}

Entities (format as JSON with categories):"""
    
    def __init__(
        self,
        provider: str = "deepseek",
        model: str = "2656053fa69c4c2d89c5a691d9d737c3",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 300,
    ):
        """
        Initialize metadata generator.
        
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
        self.enable_thinking = parse_optional_bool_env("RAG_LLM_ENABLE_THINKING")
        
        # Get API credentials
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "6fcecb364d0647d2883e7f1d3f19d5b9")
        self.base_url = base_url or "https://wishub-x6.ctyun.cn/v1"
        
        # Initialize LLM
        llm_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        extra_body = build_extra_body(enable_thinking=self.enable_thinking)
        if extra_body:
            llm_kwargs["extra_body"] = extra_body

        self.llm = ChatOpenAI(**llm_kwargs)
    
    def generate_summary(self, text: str) -> str:
        """
        Generate summary for text.
        
        Args:
            text: Input text
            
        Returns:
            Generated summary
        """
        prompt = self.SUMMARY_PROMPT.format(text=text)
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Summary generation failed: {e}")
            return text[:200] if len(text) > 200 else text
    
    def generate_keywords(self, text: str) -> List[str]:
        """
        Generate keywords for text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        prompt = self.KEYWORDS_PROMPT.format(text=text)
        
        try:
            response = self.llm.invoke(prompt)
            keywords_str = response.content.strip()
            
            # Parse comma-separated list
            keywords = [k.strip() for k in keywords_str.split(',')]
            keywords = [k for k in keywords if k]
            
            return keywords
        except Exception as e:
            print(f"Keyword generation failed: {e}")
            return []
    
    def generate_questions(self, text: str) -> List[str]:
        """
        Generate questions that text could answer.
        
        Args:
            text: Input text
            
        Returns:
            List of questions
        """
        prompt = self.QUESTIONS_PROMPT.format(text=text)
        
        try:
            response = self.llm.invoke(prompt)
            questions_str = response.content.strip()
            
            # Parse questions (split by newlines or numbers)
            questions = []
            for line in questions_str.split('\n'):
                line = line.strip()
                if line and ('?' in line or line[0].isdigit()):
                    # Remove numbering
                    if line[0].isdigit():
                        line = line[2:].strip()
                    questions.append(line)
            
            return questions
        except Exception as e:
            print(f"Question generation failed: {e}")
            return []
    
    def generate_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity categories to lists
        """
        prompt = self.ENTITIES_PROMPT.format(text=text)
        
        try:
            response = self.llm.invoke(prompt)
            entities_str = response.content.strip()
            
            # Try to parse as JSON
            try:
                # Clean up potential JSON
                if entities_str.startswith('```json'):
                    entities_str = entities_str[7:]
                if entities_str.endswith('```'):
                    entities_str = entities_str[:-3]
                
                entities = json.loads(entities_str)
                return entities
            except:
                # Fallback: simple keyword extraction
                return {'entities': entities_str.split('\n')}
        except Exception as e:
            print(f"Entity extraction failed: {e}")
            return {}
    
    def generate_all(
        self,
        text: str,
        include_summary: bool = True,
        include_keywords: bool = True,
        include_questions: bool = True,
        include_entities: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate all metadata for text.
        
        Args:
            text: Input text
            include_summary: Generate summary
            include_keywords: Generate keywords
            include_questions: Generate questions
            include_entities: Generate entities
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        if include_summary:
            metadata['summary'] = self.generate_summary(text)
        
        if include_keywords:
            metadata['keywords'] = self.generate_keywords(text)
        
        if include_questions:
            metadata['questions'] = self.generate_questions(text)
        
        if include_entities:
            metadata['entities'] = self.generate_entities(text)
        
        return metadata


class RuleBasedMetadataGenerator:
    """
    Rule-based metadata generator (no LLM)
    
    Uses simple NLP techniques:
    - Extractive summarization (first N sentences)
    - TF-IDF keyword extraction
    - Pattern-based entity extraction
    """
    
    # Medical entity patterns
    DISEASE_PATTERNS = [
        r'\b[A-Z][a-z]+ (?:syndrome|disease|disorder|condition)\b',
        r'\b(?:acute|chronic|severe) \b\w+\b',
    ]
    
    MEDICATION_PATTERNS = [
        r'\b[A-Z][a-z]+(?:ol|ine|ase|in)\b',
        r'\b(?:ACE inhibitor|beta blocker|diuretic|antibiotic)\b',
    ]
    
    def __init__(
        self,
        summary_sentences: int = 2,
        max_keywords: int = 5,
    ):
        """
        Initialize rule-based generator.
        
        Args:
            summary_sentences: Number of sentences for summary
            max_keywords: Maximum keywords to extract
        """
        self.summary_sentences = summary_sentences
        self.max_keywords = max_keywords
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def generate_summary(self, text: str) -> str:
        """Generate extractive summary (first N sentences)"""
        sentences = self._split_sentences(text)
        summary = sentences[:self.summary_sentences]
        return ' '.join(summary)
    
    def generate_keywords(self, text: str) -> List[str]:
        """Extract keywords using simple frequency analysis"""
        import re
        from collections import Counter
        
        # Tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
            'neither', 'not', 'only', 'own', 'same', 'than', 'too',
            'very', 'just', 'also', 'now', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'them', 'their', 'which',
            'who', 'whom', 'whose', 'what', 'whatever', 'when', 'where',
            'why', 'how', 'all', 'each', 'every', 'any', 'some', 'no',
        }
        
        filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        
        # Get top keywords
        keywords = [word for word, count in word_counts.most_common(self.max_keywords)]
        
        return keywords
    
    def generate_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using pattern matching"""
        import re
        
        entities = {
            'diseases': [],
            'medications': [],
        }
        
        # Extract diseases
        for pattern in self.DISEASE_PATTERNS:
            matches = re.findall(pattern, text)
            entities['diseases'].extend(matches)
        
        # Extract medications
        for pattern in self.MEDICATION_PATTERNS:
            matches = re.findall(pattern, text)
            entities['medications'].extend(matches)
        
        # Remove duplicates
        entities['diseases'] = list(set(entities['diseases']))
        entities['medications'] = list(set(entities['medications']))
        
        return entities
    
    def generate_all(
        self,
        text: str,
        include_summary: bool = True,
        include_keywords: bool = True,
        include_entities: bool = True,
    ) -> Dict[str, Any]:
        """Generate all metadata"""
        metadata = {}
        
        if include_summary:
            metadata['summary'] = self.generate_summary(text)
        
        if include_keywords:
            metadata['keywords'] = self.generate_keywords(text)
        
        if include_entities:
            metadata['entities'] = self.generate_entities(text)
        
        return metadata


class MetadataEnhancedChunker:
    """
    Chunking with automatic metadata enhancement
    
    Combines chunking with metadata generation
    """
    
    def __init__(
        self,
        chunker=None,
        metadata_generator=None,
        use_llm: bool = True,
    ):
        """
        Initialize enhanced chunker.
        
        Args:
            chunker: Chunking strategy to use
            metadata_generator: Metadata generator to use
            use_llm: Use LLM for metadata (True) or rules (False)
        """
        from chunking import SemanticChunker
        
        self.chunker = chunker or SemanticChunker()
        
        if metadata_generator:
            self.metadata_generator = metadata_generator
        elif use_llm:
            self.metadata_generator = MetadataGenerator()
        else:
            self.metadata_generator = RuleBasedMetadataGenerator()
        
        self.use_llm = use_llm
    
    def chunk_and_enrich(
        self,
        text: str,
        base_metadata: Optional[Dict[str, Any]] = None,
        generate_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Chunk text and enrich with metadata.
        
        Args:
            text: Input text
            base_metadata: Base metadata to include
            generate_metadata: Whether to generate metadata
            
        Returns:
            List of chunks with metadata
        """
        from chunking import Chunk
        
        base_metadata = base_metadata or {}
        
        # Step 1: Chunk the text
        chunks = self.chunker.chunk_text(text, metadata=base_metadata)
        
        # Step 2: Generate metadata for each chunk
        if generate_metadata:
            for chunk in chunks:
                chunk_metadata = self.metadata_generator.generate_all(
                    chunk.content,
                    include_summary=True,
                    include_keywords=True,
                    include_questions=True,
                    include_entities=False,  # Skip for speed
                )
                
                # Merge with base metadata
                chunk.metadata.update(chunk_metadata)
        
        # Convert to dictionaries
        chunk_dicts = []
        for chunk in chunks:
            chunk_dict = {
                'content': chunk.content,
                'metadata': chunk.metadata,
                'chunk_id': chunk.chunk_id,
                'parent_id': chunk.parent_id,
            }
            chunk_dicts.append(chunk_dict)
        
        return chunk_dicts


if __name__ == "__main__":
    # Test metadata generation
    print("Testing Metadata Enhancement Module...")
    
    sample_text = """
    Hypertension is a common medical condition characterized by elevated blood pressure in the arteries. 
    It is often referred to as high blood pressure. Hypertension is defined as having a systolic blood 
    pressure of 130 mmHg or higher, or a diastolic blood pressure of 80 mmHg or higher.
    
    There are two types of hypertension: Primary (essential) hypertension and Secondary hypertension.
    Primary hypertension develops gradually over many years and has no identifiable cause.
    Secondary hypertension is caused by an underlying condition such as kidney disease or thyroid problems.
    
    Treatment options include lifestyle modifications and medications such as diuretics, ACE inhibitors, 
    ARBs, calcium channel blockers, and beta blockers.
    """
    
    # Test LLM-based generation
    print("\n=== LLM-based Metadata Generation ===")
    llm_generator = MetadataGenerator()
    
    metadata = llm_generator.generate_all(
        sample_text,
        include_summary=True,
        include_keywords=True,
        include_questions=True,
        include_entities=False,
    )
    
    print("\nGenerated Metadata:")
    print(f"Summary: {metadata.get('summary', 'N/A')[:200]}...")
    print(f"Keywords: {metadata.get('keywords', [])}")
    print(f"Questions: {metadata.get('questions', [])}")
    
    # Test rule-based generation
    print("\n=== Rule-based Metadata Generation ===")
    rule_generator = RuleBasedMetadataGenerator()
    
    metadata = rule_generator.generate_all(
        sample_text,
        include_summary=True,
        include_keywords=True,
        include_entities=True,
    )
    
    print("\nGenerated Metadata:")
    print(f"Summary: {metadata.get('summary', 'N/A')[:200]}...")
    print(f"Keywords: {metadata.get('keywords', [])}")
    print(f"Entities: {metadata.get('entities', {})}")
