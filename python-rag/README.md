# Medical RAG System with LangChain

A Retrieval-Augmented Generation (RAG) system for medical diagnosis support, built with Python and LangChain.

## 🎯 Project Overview

This project implements a RAG system specifically designed for medical diagnosis, following best practices from recent academic literature including MedRAG, RAGMed, and related papers.

### Key Features

- **Real Datasets**: PubMedQA, MedQA, MedMCQA
- **Real LLM**: OpenAI GPT-4o-mini (configurable)
- **Medical Embeddings**: Bio_ClinicalBERT or OpenAI embeddings
- **Comprehensive Evaluation**: RAGAS + Medical-specific metrics
- **LangChain Framework**: Modular, production-ready architecture

## 📊 Evaluation Datasets

| Dataset | Size | Task | Source |
|---------|------|------|--------|
| PubMedQA | 1,000 QA | Yes/No/Maybe reasoning | PubMed abstracts |
| MedQA | 12,723 questions | USMLE multiple choice | Medical exams |
| MedMCQA | 194,000+ questions | Multiple choice | AIIMS/NEET exams |

## 🛠 Installation

### Prerequisites

- Python 3.10+
- OpenAI API key (or other LLM API)

### Setup

```bash
# Clone or download the project
cd python-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-api-key-here"
```

## 🚀 Quick Start

### Basic Usage

```python
from app.rag.improved_medical_rag import MedicalRAGSystem, MedicalRAGConfig

# Configure the system
config = MedicalRAGConfig(
    llm_model="gpt-4o-mini",
    llm_temperature=0.1,
    top_k=5
)

# Initialize
system = MedicalRAGSystem(config)
system.initialize()

# Query
result = system.query("What are the treatments for hypertension?")
print(result['answer'])
```

### Running the API Server

```bash
# Start FastAPI server
cd python-rag
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Access API documentation
# http://localhost:8000/docs
```

## 📁 Project Structure

```
python-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   └── rag/
│       ├── __init__.py
│       ├── config.py              # Configuration
│       ├── document_loader.py     # Document processing
│       ├── embeddings.py          # Embedding models
│       ├── vector_store.py        # Vector store (FAISS)
│       ├── pipeline.py            # RAG pipeline
│       ├── improved_medical_rag.py # Improved implementation
│       └── evaluation.py          # Evaluation module
├── scripts/
│   ├── start.sh                   # Startup script
│   └── test_system.py             # Test script
├── data/                          # Data directory
├── requirements.txt
└── README.md
```

## 📈 Evaluation

### RAGAS Metrics

| Metric | Description |
|--------|-------------|
| Faithfulness | Answer grounded in retrieved context |
| Answer Relevancy | Answer addresses the question |
| Context Relevancy | Retrieved context is relevant |
| Context Precision | Precision of retrieved context |

### Medical-Specific Metrics

| Metric | Description |
|--------|-------------|
| Medical Accuracy | Correctness of medical information |
| Safety Score | Presence of disclaimers, absence of dangerous advice |
| Completeness | Coverage of required medical elements |

### Running Evaluation

```python
from app.rag.evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator()

results = evaluator.evaluate(
    question="What is the treatment for MI?",
    answer="...",
    contexts=["..."],
    ground_truth="..."
)

print(f"Overall Score: {results['overall_score']}")
print(f"Safety Score: {results['safety']['score']}")
```

## 🔧 Configuration

### LLM Options

| Model | Provider | Notes |
|-------|----------|-------|
| gpt-4o-mini | OpenAI | Recommended (cost-effective) |
| gpt-4o | OpenAI | Best performance |
| glm-4 | Zhipu AI | Chinese language support |

### Embedding Options

| Model | Dimensions | Notes |
|-------|------------|-------|
| text-embedding-3-small | 1536 | OpenAI, recommended |
| Bio_ClinicalBERT | 768 | Medical domain-specific |
| all-MiniLM-L6-v2 | 384 | Lightweight, fast |

## 📚 References

### Key Papers

1. **MedRAG** (2025): Knowledge graph-enhanced medical RAG
2. **RAGMed** (2025): RAG-based medical AI assistant
3. **RAG in Healthcare Survey** (2025): Comprehensive review

### Datasets

- [PubMedQA](https://huggingface.co/datasets/pubmed_qa)
- [MedQA](https://github.com/jind11/MedQA)
- [MedMCQA](https://medmcqa.github.io/)

## ⚠️ Important Notes

1. **API Keys**: This project requires an OpenAI API key (or compatible LLM API)
2. **Medical Disclaimer**: This system is for educational purposes only. Always consult healthcare professionals for medical advice.
3. **Evaluation**: For accurate evaluation, use the full PubMedQA/MedQA datasets

## 📝 License

This project is for educational purposes as part of a university course project.

## 👤 Author

Sun Baozheng (59433383)
