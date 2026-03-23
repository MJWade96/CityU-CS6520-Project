"""
Document Chunking Module

Implements advanced chunking strategies:
1. Semantic chunking (based on sentence boundaries and paragraphs)
2. Sliding window (overlapping chunks)
3. Parent-Child association (fine-grained retrieval, coarse-grained generation)
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib


@dataclass
class Chunk:
    """Represents a document chunk"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    parent_id: Optional[str] = None
    start_idx: int = 0
    end_idx: int = 0
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]
        return f"chunk_{content_hash}"


class SemanticChunker:
    """
    Semantic chunker that splits text based on meaning
    
    Strategies:
    - Paragraph-based splitting
    - Sentence boundary detection
    - Semantic similarity between sentences
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1024,
        sentence_separator: str = r'(?<=[.!?])\s+',
    ):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Target chunk size in tokens/characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            sentence_separator: Regex pattern for sentence splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.sentence_separator = sentence_separator
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(self.sentence_separator, text)
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count (4 characters per token)"""
        return len(text) // 4
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk text using semantic boundaries.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Step 1: Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = ""
        current_length = 0
        start_idx = 0
        
        for para in paragraphs:
            para_length = self._count_tokens(para)
            
            # If paragraph itself is too long, split into sentences
            if para_length > self.max_chunk_size:
                sentences = self._split_into_sentences(para)
                
                for sentence in sentences:
                    sent_length = self._count_tokens(sentence)
                    
                    if current_length + sent_length > self.chunk_size:
                        # Save current chunk
                        if current_length >= self.min_chunk_size:
                            chunk = Chunk(
                                content=current_chunk.strip(),
                                metadata=metadata.copy(),
                                start_idx=start_idx,
                                end_idx=start_idx + len(current_chunk),
                            )
                            chunks.append(chunk)
                        
                        # Start new chunk with overlap
                        overlap_text = self._get_overlap_text(current_chunk)
                        current_chunk = overlap_text + " " + sentence
                        start_idx = max(0, start_idx + len(current_chunk) - len(overlap_text) - len(sentence))
                        current_length = self._count_tokens(current_chunk)
                    else:
                        current_chunk += " " + sentence
                        current_length += sent_length
            else:
                # Add paragraph to current chunk
                if current_length + para_length > self.chunk_size:
                    # Save current chunk
                    if current_length >= self.min_chunk_size:
                        chunk = Chunk(
                            content=current_chunk.strip(),
                            metadata=metadata.copy(),
                            start_idx=start_idx,
                            end_idx=start_idx + len(current_chunk),
                        )
                        chunks.append(chunk)
                    
                    # Start new chunk
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + para
                    start_idx = max(0, len(current_chunk) - len(overlap_text) - len(para))
                    current_length = self._count_tokens(current_chunk)
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                    current_length += para_length
        
        # Add final chunk
        if current_chunk and self._count_tokens(current_chunk) >= self.min_chunk_size:
            chunk = Chunk(
                content=current_chunk.strip(),
                metadata=metadata.copy(),
                start_idx=start_idx,
                end_idx=start_idx + len(current_chunk),
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of previous chunk"""
        if not text or self.chunk_overlap == 0:
            return ""
        
        # Try to split at sentence boundary
        sentences = self._split_into_sentences(text)
        
        overlap = ""
        overlap_length = 0
        
        for sentence in reversed(sentences):
            if overlap_length + self._count_tokens(sentence) <= self.chunk_overlap:
                overlap = sentence + " " + overlap
                overlap_length += self._count_tokens(sentence)
            else:
                break
        
        if not overlap:
            # Fallback: use last chunk_overlap characters
            overlap = text[-self.chunk_overlap:] if len(text) > self.chunk_overlap else text
        
        return overlap.strip()


class ParentChildChunker:
    """
    Parent-Child chunking strategy
    
    Creates two levels of chunks:
    - Parent chunks: Large, for context during generation (1000-2000 tokens)
    - Child chunks: Small, for retrieval (100-200 tokens)
    
    Child chunks reference their parent via parent_id
    """
    
    def __init__(
        self,
        parent_chunk_size: int = 1024,
        child_chunk_size: int = 200,
        child_overlap: int = 50,
        min_child_size: int = 100,
    ):
        """
        Initialize parent-child chunker.
        
        Args:
            parent_chunk_size: Size of parent chunks
            child_chunk_size: Size of child chunks
            child_overlap: Overlap between child chunks
            min_child_size: Minimum child chunk size
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.min_child_size = min_child_size
        
        self.parent_chunker = SemanticChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=0,
            min_chunk_size=200,
        )
        
        self.child_chunker = SemanticChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            min_chunk_size=min_child_size,
        )
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Create parent and child chunks.
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        metadata = metadata or {}
        
        # Step 1: Create parent chunks
        parent_chunks = self.parent_chunker.chunk_text(text, metadata)
        
        # Step 2: For each parent, create child chunks
        child_chunks = []
        
        for parent in parent_chunks:
            # Create child chunks from parent content
            children = self.child_chunker.chunk_text(
                parent.content,
                metadata=metadata.copy()
            )
            
            # Set parent-child relationship
            for child in children:
                child.parent_id = parent.chunk_id
                child.metadata['parent_id'] = parent.chunk_id
                child.metadata['parent_content'] = parent.content
            
            child_chunks.extend(children)
        
        return parent_chunks, child_chunks


class SlidingWindowChunker:
    """
    Simple sliding window chunker
    
    Creates chunks with fixed size and overlap
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize sliding window chunker.
        
        Args:
            chunk_size: Chunk size in characters
            chunk_overlap: Overlap in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk text using sliding window.
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []
        
        metadata = metadata or {}
        chunks = []
        
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to split at sentence boundary
            if end < len(text):
                # Find last sentence boundary
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                
                split_point = max(last_period, last_newline)
                
                if split_point > start:
                    end = split_point + 1
            
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunk = Chunk(
                    content=chunk_text.strip(),
                    metadata=metadata.copy(),
                    chunk_id=f"chunk_{chunk_idx}_{start}",
                    start_idx=start,
                    end_idx=end,
                )
                chunks.append(chunk)
                chunk_idx += 1
            
            # Move start with overlap
            start = end - self.chunk_overlap
        
        return chunks


class ChunkingPipeline:
    """
    Complete chunking pipeline
    
    Supports:
    - Semantic chunking
    - Parent-child chunking
    - Sliding window chunking
    """
    
    def __init__(
        self,
        strategy: str = 'semantic',
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_parent_child: bool = False,
    ):
        """
        Initialize chunking pipeline.
        
        Args:
            strategy: Chunking strategy ('semantic', 'sliding_window', 'parent_child')
            chunk_size: Base chunk size
            chunk_overlap: Chunk overlap
            use_parent_child: Use parent-child chunking
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_parent_child = use_parent_child
        
        # Initialize chunkers
        if strategy == 'semantic':
            self.chunker = SemanticChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif strategy == 'sliding_window':
            self.chunker = SlidingWindowChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif strategy == 'parent_child' or use_parent_child:
            self.chunker = ParentChildChunker(
                parent_chunk_size=chunk_size * 2,
                child_chunk_size=chunk_size // 2,
                child_overlap=chunk_overlap,
            )
        else:
            self.chunker = SemanticChunker()
    
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk text.
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            List of chunks
        """
        if isinstance(self.chunker, ParentChildChunker):
            parent_chunks, child_chunks = self.chunker.chunk_text(text, metadata)
            # Return child chunks for retrieval (with parent references)
            return child_chunks
        else:
            return self.chunker.chunk_text(text, metadata)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'strategy': self.strategy,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'use_parent_child': self.use_parent_child,
        }


if __name__ == "__main__":
    # Test chunking strategies
    print("Testing Document Chunking Module...")
    
    # Sample text
    sample_text = """
    Hypertension is a common medical condition characterized by elevated blood pressure in the arteries. 
    It is often referred to as high blood pressure.
    
    Hypertension is defined as having a systolic blood pressure of 130 mmHg or higher, or a diastolic blood pressure of 80 mmHg or higher.
    
    There are two types of hypertension:
    1. Primary (essential) hypertension: This is the most common type, accounting for 90-95% of cases. It develops gradually over many years and has no identifiable cause.
    2. Secondary hypertension: This type is caused by an underlying condition such as kidney disease, thyroid problems, or certain medications.
    
    Risk factors for hypertension include:
    - Age: The risk increases as you get older
    - Family history: Genetics play a role
    - Obesity: Being overweight increases risk
    - Physical inactivity: Lack of exercise contributes
    - Tobacco use: Smoking or chewing tobacco
    - High salt intake: Excessive sodium in diet
    
    Complications of untreated hypertension include:
    - Heart attack and heart failure
    - Stroke
    - Kidney damage
    - Vision loss
    - Sexual dysfunction
    - Peripheral artery disease
    
    Treatment options include lifestyle modifications and medications.
    Lifestyle changes include weight loss, regular exercise, healthy diet, and reducing salt intake.
    Medications include diuretics, ACE inhibitors, ARBs, calcium channel blockers, and beta blockers.
    """
    
    # Test semantic chunking
    print("\n=== Semantic Chunking ===")
    semantic_chunker = SemanticChunker(chunk_size=200, chunk_overlap=30)
    chunks = semantic_chunker.chunk_text(sample_text, metadata={"source": "test"})
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} (ID: {chunk.chunk_id}):")
        print(f"  Length: {len(chunk.content)} chars")
        print(f"  Content: {chunk.content[:100]}...")
    
    # Test sliding window
    print("\n=== Sliding Window Chunking ===")
    sliding_chunker = SlidingWindowChunker(chunk_size=300, chunk_overlap=50)
    chunks = sliding_chunker.chunk_text(sample_text)
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Length: {len(chunk.content)} chars")
        print(f"  Content: {chunk.content[:100]}...")
    
    # Test parent-child chunking
    print("\n=== Parent-Child Chunking ===")
    pc_chunker = ParentChildChunker(
        parent_chunk_size=400,
        child_chunk_size=150,
        child_overlap=30
    )
    parent_chunks, child_chunks = pc_chunker.chunk_text(sample_text)
    
    print(f"Created {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks:")
    for i, parent in enumerate(parent_chunks, 1):
        print(f"\nParent {i} (ID: {parent.chunk_id}):")
        print(f"  Length: {len(parent.content)} chars")
        
        # Find children
        children = [c for c in child_chunks if c.parent_id == parent.chunk_id]
        print(f"  Children: {len(children)}")
        
        for j, child in enumerate(children, 1):
            print(f"    Child {j}: {child.content[:50]}...")
