"""
Medical Corpus Loader Module

Handles loading and processing of medical corpora:
1. StatPearls - Medical encyclopedia (paragraph-level chunking)
2. PubMed Abstracts - Research abstracts (full abstract as chunk)

Data sources should be placed in:
- {CORPUS_DIR}/statpearls/
- {CORPUS_DIR}/pubmed/
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from .data_paths import CORPUS_DIR
from .statpearls_dataset import build_statpearls_dataset


@dataclass
class ChunkedDocument:
    """Document with chunking metadata"""
    doc_id: str
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    token_count: Optional[int] = None


class StatPearlsLoader:
    """
    Loader for StatPearls medical encyclopedia.

    Processing strategy:
    - Chunk by Markdown headers or paragraphs
    - Target chunk size: 256-512 tokens
    """

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""
            ],
            keep_separator=True
        )

    def load_directory(self, directory: str) -> List[ChunkedDocument]:
        """Load all markdown files from directory"""
        documents = []
        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        for file_path in directory_path.rglob("*.md"):
            chunks = self.load_file(str(file_path))
            documents.extend(chunks)

        print(f"Loaded {len(documents)} chunks from {directory}")
        return documents

    def load_file(self, file_path: str) -> List[ChunkedDocument]:
        """Load and chunk a single markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        file_name = Path(file_path).stem

        splits = self.text_splitter.split_text(content)

        chunks = []
        for i, split in enumerate(splits):
            chunks.append(ChunkedDocument(
                doc_id=file_name,
                chunk_id=f"{file_name}_chunk_{i}",
                content=split,
                metadata={
                    'source': 'statpearls',
                    'file': file_path,
                    'title': file_name,
                    'chunk_index': i,
                    'total_chunks': len(splits)
                }
            ))

        return chunks


class PubMedLoader:
    """
    Loader for PubMed abstracts.

    Processing strategy:
    - Each abstract as one chunk (typically short)
    - Keep metadata: PMID, title, authors, journal
    """

    def __init__(self):
        pass

    def load_json(self, file_path: str) -> List[ChunkedDocument]:
        """Load PubMed abstracts from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'records' in data:
            records = data['records']
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError("Invalid PubMed JSON format")

        documents = []
        for i, record in enumerate(records):
            doc = self._parse_record(record, i)
            if doc:
                documents.append(doc)

        print(f"Loaded {len(documents)} PubMed abstracts")
        return documents

    def _parse_record(self, record: Dict, index: int) -> Optional[ChunkedDocument]:
        """Parse a single PubMed record"""
        pmid = record.get('pmid', str(index))
        title = record.get('title', '')
        abstract = record.get('abstract', '')

        if not abstract:
            return None

        content = f"{title}\n\n{abstract}"

        return ChunkedDocument(
            doc_id=pmid,
            chunk_id=f"{pmid}_chunk_0",
            content=content,
            metadata={
                'source': 'pubmed',
                'pmid': pmid,
                'title': title,
                'authors': record.get('authors', []),
                'journal': record.get('journal', ''),
                'year': record.get('year', ''),
                'mesh_terms': record.get('mesh_terms', [])
            }
        )

    def load_csv(self, file_path: str) -> List[ChunkedDocument]:
        """Load PubMed abstracts from CSV file"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for CSV loading")

        df = pd.read_csv(file_path)

        documents = []
        for i, row in df.iterrows():
            if pd.isna(row.get('abstract', '')):
                continue

            pmid = str(row.get('pmid', i))

            documents.append(ChunkedDocument(
                doc_id=pmid,
                chunk_id=f"{pmid}_chunk_0",
                content=f"{row.get('title', '')}\n\n{row.get('abstract', '')}",
                metadata={
                    'source': 'pubmed',
                    'pmid': pmid,
                    'title': row.get('title', ''),
                    'authors': row.get('authors', ''),
                    'journal': row.get('journal', ''),
                    'year': row.get('year', '')
                }
            ))

        print(f"Loaded {len(documents)} PubMed abstracts from CSV")
        return documents


class MedicalCorpusLoader:
    """
    Unified loader for all medical corpora.
    Supports StatPearls and PubMed.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.statpearls_loader = StatPearlsLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.pubmed_loader = PubMedLoader()

    def load_corpus(
        self,
        corpus_path: str = None,
        sources: List[str] = None
    ) -> List[ChunkedDocument]:
        """
        Load corpus from specified sources.

        Args:
            corpus_path: Base path for corpus data
            sources: List of sources to load ('statpearls', 'pubmed')

        Returns:
            List of chunked documents
        """
        if corpus_path is None:
            corpus_path = self._get_default_corpus_path()

        if sources is None:
            sources = ['statpearls', 'pubmed']

        all_documents = []

        for source in sources:
            source_path = os.path.join(corpus_path, source)

            if not os.path.exists(source_path):
                print(f"Warning: Source path not found: {source_path}")
                continue

            if source == 'statpearls':
                docs = self.statpearls_loader.load_directory(source_path)
            elif source == 'pubmed':
                for file in Path(source_path).glob("*.json"):
                    docs = self.pubmed_loader.load_json(str(file))
                    all_documents.extend(docs)
                continue
            else:
                print(f"Unknown source: {source}")
                continue

            all_documents.extend(docs)

        return all_documents

    def _get_default_corpus_path(self) -> str:
        """Return the configured corpus root."""
        return str(CORPUS_DIR)

    def chunk_to_langchain_docs(
        self,
        chunks: List[ChunkedDocument]
    ) -> List[Document]:
        """Convert chunked documents to LangChain Documents"""
        return [
            Document(
                page_content=chunk.content,
                metadata=chunk.metadata
            )
            for chunk in chunks
        ]


def download_statpearls(output_dir: str) -> None:
    """
    Download StatPearls data using the shared downloader.
    """
    result = build_statpearls_dataset(Path(output_dir).parent)
    print(f"Downloaded StatPearls to: {result['combined_file']}")


def download_pubmed(
    output_file: str,
    query: str = "medical[Title/Abstract]",
    max_results: int = 1000
) -> None:
    """
    Download PubMed abstracts using Entrez API.

    Args:
        output_file: Output JSON file path
        query: PubMed search query
        max_results: Maximum number of results
    """
    try:
        from Bio import Entrez
    except ImportError:
        raise ImportError("Biopython required. Install: pip install biopython")

    Entrez.email = "your_email@example.com"

    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance"
    )
    search_results = Entrez.read(handle)

    id_list = search_results["IdList"]

    handle = Entrez.efetch(
        db="pubmed",
        id=id_list,
        rettype="abstract",
        retmode="json"
    )
    records = Entrez.read(handle)

    records_list = []
    for pmid in id_list:
        try:
            abstract_info = records['PubmedArticle'][pmid]['MedlineCitation']['Article']
            record = {
                'pmid': pmid,
                'title': abstract_info.get('ArticleTitle', ''),
                'abstract': abstract_info.get('Abstract', {}).get('AbstractText', [''])[0],
                'authors': [a.get('LastName', '') for a in abstract_info.get('AuthorList', [])],
                'journal': abstract_info.get('Journal', {}).get('Title', ''),
                'year': abstract_info.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {}).get('Year', '')
            }
            records_list.append(record)
        except:
            continue

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(records_list, f, ensure_ascii=False, indent=2)

    print(f"Downloaded {len(records_list)} PubMed abstracts to {output_file}")


def demo():
    """Demonstrate corpus loading"""
    print("=" * 60)
    print("Medical Corpus Loader Demo")
    print("=" * 60)

    print("\nTo use the corpus loader:")
    print(f"1. Place StatPearls markdown files in: {CORPUS_DIR / 'statpearls'}")
    print(f"2. Place PubMed JSON files in: {CORPUS_DIR / 'pubmed'}")
    print("\nExample usage:")
    print("""
    from app.rag.corpus_loader import MedicalCorpusLoader

    loader = MedicalCorpusLoader(chunk_size=512, chunk_overlap=50)

    # Load corpus
    chunks = loader.load_corpus(
        corpus_path=None,  # uses app.rag.data_paths.CORPUS_DIR by default
        sources=['statpearls', 'pubmed']
    )

    # Convert to LangChain documents
    docs = loader.chunk_to_langchain_docs(chunks)
    """)


if __name__ == "__main__":
    demo()
