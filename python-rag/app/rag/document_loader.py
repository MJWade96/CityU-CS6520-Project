"""
Document Loader Module
Handles loading and preprocessing medical documents using LangChain
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)


@dataclass
class ProcessedDocument:
    """Processed document with metadata"""

    id: str
    content: str
    source: str
    title: str
    category: str
    metadata: Dict[str, Any]


class MedicalDocumentLoader:
    """
    Medical Document Loader using LangChain

    Handles loading, preprocessing, and chunking of medical documents
    from various sources including PDFs, text files, and markdown.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize the document loader.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: Custom separators for text splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators optimized for medical text
        self.separators = separators or [
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            ". ",  # Sentence endings
            " ",  # Word boundaries
            "",  # Character level
        ]

        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

        # Token-based splitter for more precise chunking
        self.token_splitter = TokenTextSplitter(
            chunk_size=chunk_size // 4,  # Approximate tokens
            chunk_overlap=chunk_overlap // 4,
        )

    def load_text_file(self, file_path: str) -> List[Document]:
        """Load a text file and return LangChain Documents"""
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    def load_pdf_file(self, file_path: str) -> List[Document]:
        """Load a PDF file and return LangChain Documents"""
        loader = PyPDFLoader(file_path)
        return loader.load()

    def load_markdown_file(self, file_path: str) -> List[Document]:
        """Load a markdown file and return LangChain Documents"""
        loader = UnstructuredMarkdownLoader(file_path)
        return loader.load()

    def load_directory(
        self, directory_path: str, glob_pattern: str = "**/*.txt"
    ) -> List[Document]:
        """Load all matching files from a directory"""
        loader = DirectoryLoader(directory_path, glob=glob_pattern, show_progress=True)
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of chunked Document objects
        """
        return self.text_splitter.split_documents(documents)

    def process_medical_document(
        self, content: str, metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Process a medical document with custom metadata.

        Args:
            content: The document text content
            metadata: Document metadata (source, title, category, etc.)

        Returns:
            List of chunked Document objects with metadata
        """
        # Create a LangChain Document
        doc = Document(page_content=content, metadata=metadata)

        # Split into chunks
        chunks = self.text_splitter.split_documents([doc])

        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def preprocess_medical_text(self, text: str) -> str:
        """
        Preprocess medical text for better retrieval.

        - Normalize whitespace
        - Handle medical abbreviations
        - Clean special characters
        """
        # Normalize whitespace
        text = " ".join(text.split())

        # Common medical abbreviation expansion
        abbreviations = {
            "MI": "myocardial infarction",
            "HTN": "hypertension",
            "DM": "diabetes mellitus",
            "CHF": "congestive heart failure",
            "COPD": "chronic obstructive pulmonary disease",
            "CVA": "cerebrovascular accident",
            "Dx": "diagnosis",
            "Tx": "treatment",
            "Hx": "history",
            "Px": "prognosis",
            "Rx": "prescription",
            "BID": "twice daily",
            "TID": "three times daily",
            "QID": "four times daily",
            "PRN": "as needed",
        }

        # Expand abbreviations (case-sensitive)
        words = text.split()
        expanded_words = []
        for word in words:
            # Check if word is an abbreviation (with punctuation handling)
            clean_word = word.strip(".,;:!?")
            if clean_word in abbreviations:
                expanded_words.append(abbreviations[clean_word])
            else:
                expanded_words.append(word)

        return " ".join(expanded_words)


class MedicalKnowledgeBase:
    """
    Medical Knowledge Base Manager

    Manages the collection of medical documents for the RAG system.
    """

    def __init__(self, loader: Optional[MedicalDocumentLoader] = None):
        """Initialize the knowledge base"""
        self.loader = loader or MedicalDocumentLoader()
        self.documents: List[Document] = []
        self.metadata: Dict[str, Any] = {}

    def add_document(
        self,
        content: str,
        source: str,
        title: str,
        category: str,
        **additional_metadata,
    ) -> None:
        """Add a document to the knowledge base"""
        metadata = {
            "source": source,
            "title": title,
            "category": category,
            **additional_metadata,
        }

        # Preprocess and chunk
        preprocessed = self.loader.preprocess_medical_text(content)
        chunks = self.loader.process_medical_document(preprocessed, metadata)

        self.documents.extend(chunks)

    def load_from_directory(self, directory_path: str) -> None:
        """Load all documents from a directory"""
        path = Path(directory_path)

        # Load PDFs
        for pdf_file in path.glob("**/*.pdf"):
            docs = self.loader.load_pdf_file(str(pdf_file))
            chunks = self.loader.split_documents(docs)
            self.documents.extend(chunks)

        # Load text files
        for txt_file in path.glob("**/*.txt"):
            docs = self.loader.load_text_file(str(txt_file))
            chunks = self.loader.split_documents(docs)
            self.documents.extend(chunks)

        # Load markdown files
        for md_file in path.glob("**/*.md"):
            docs = self.loader.load_markdown_file(str(md_file))
            chunks = self.loader.split_documents(docs)
            self.documents.extend(chunks)

    def get_documents(self) -> List[Document]:
        """Get all documents in the knowledge base"""
        return self.documents

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        categories = set()
        sources = set()

        for doc in self.documents:
            if "category" in doc.metadata:
                categories.add(doc.metadata["category"])
            if "source" in doc.metadata:
                sources.add(doc.metadata["source"])

        return {
            "total_documents": len(self.documents),
            "categories": list(categories),
            "sources": list(sources),
        }

    def clear(self) -> None:
        """Clear all documents from the knowledge base"""
        self.documents = []
        self.metadata = {}


# Sample medical knowledge for demonstration
SAMPLE_MEDICAL_KNOWLEDGE = [
    {
        "content": """Hypertension Clinical Practice Guidelines

Hypertension, commonly known as high blood pressure, is a major risk factor for cardiovascular disease, stroke, and kidney disease.

Blood Pressure Classification:
- Normal: Systolic < 120 mmHg AND Diastolic < 80 mmHg
- Elevated: Systolic 120-129 mmHg AND Diastolic < 80 mmHg
- Stage 1 Hypertension: Systolic 130-139 mmHg OR Diastolic 80-89 mmHg
- Stage 2 Hypertension: Systolic ≥ 140 mmHg OR Diastolic ≥ 90 mmHg

Treatment Approach:
Non-pharmacological interventions include weight loss, DASH diet, sodium reduction, and regular physical activity.

First-line medications include:
- Thiazide diuretics
- Calcium channel blockers
- ACE inhibitors or ARBs
- Beta-blockers (specific indications)

Target blood pressure: < 130/80 mmHg for most adults.""",
        "source": "American Heart Association",
        "title": "Hypertension Clinical Practice Guidelines 2023",
        "category": "clinical_guidelines",
    },
    {
        "content": """Type 2 Diabetes Mellitus: Diagnosis and Management

Type 2 diabetes mellitus (T2DM) is a chronic metabolic disorder characterized by hyperglycemia resulting from insulin resistance and relative insulin deficiency.

Diagnostic Criteria:
- Fasting plasma glucose ≥ 126 mg/dL (7.0 mmol/L)
- 2-hour plasma glucose ≥ 200 mg/dL during OGTT
- HbA1c ≥ 6.5%
- Random plasma glucose ≥ 200 mg/dL with classic symptoms

Management Approach:
Lifestyle modifications include medical nutrition therapy, regular physical activity (150 minutes/week), and weight management.

Pharmacological Treatment:
First-line: Metformin (starting 500 mg, max 2550 mg/day)
Second-line options: SGLT2 inhibitors, GLP-1 receptor agonists, DPP-4 inhibitors

Glycemic Targets:
- HbA1c: < 7% for most adults
- Fasting glucose: 80-130 mg/dL
- Postprandial glucose: < 180 mg/dL""",
        "source": "American Diabetes Association",
        "title": "Standards of Medical Care in Diabetes - 2024",
        "category": "clinical_guidelines",
    },
    {
        "content": """Acute Myocardial Infarction: Diagnosis and Treatment

Acute myocardial infarction (AMI), commonly known as a heart attack, occurs when there is prolonged ischemia leading to irreversible myocardial cell death.

Clinical Presentation:
- Chest pain or pressure lasting more than 20 minutes
- Pain radiating to left arm, jaw, or back
- Associated symptoms: diaphoresis, nausea, dyspnea

Diagnostic Evaluation:
ECG: ST-segment elevation ≥ 1mm in two contiguous leads (STEMI)
Cardiac Biomarkers: Elevated troponin I or T above 99th percentile

Management:
STEMI: Primary PCI within 90 minutes of first medical contact
NSTEMI: Antiplatelet therapy, anticoagulation, early invasive strategy

Medications for Secondary Prevention:
- Aspirin 81 mg daily indefinitely
- Beta-blockers (metoprolol, carvedilol)
- ACE inhibitors or ARBs
- High-intensity statins""",
        "source": "American College of Cardiology",
        "title": "ACC/AHA Guidelines for Acute Myocardial Infarction",
        "category": "clinical_guidelines",
    },
    {
        "content": """Community-Acquired Pneumonia: Diagnosis and Treatment

Community-acquired pneumonia (CAP) is an acute infection of the pulmonary parenchyma acquired outside of hospital settings.

Common Pathogens:
- Streptococcus pneumoniae (most common bacterial cause)
- Haemophilus influenzae
- Mycoplasma pneumoniae
- Respiratory viruses (influenza, RSV)

Clinical Features:
- Fever, chills, cough with purulent sputum
- Pleuritic chest pain, dyspnea
- Crackles on auscultation

Severity Assessment (CURB-65):
- Confusion
- Urea > 7 mmol/L
- Respiratory rate ≥ 30/min
- Blood pressure < 90/60 mmHg
- Age ≥ 65 years

Treatment:
Outpatient (healthy): Amoxicillin or Doxycycline
Outpatient (comorbidities): Amoxicillin-clavulanate + macrolide
Inpatient: Ceftriaxone + azithromycin

Duration: 5-7 days for uncomplicated CAP""",
        "source": "Infectious Diseases Society of America",
        "title": "IDSA/ATS Guidelines for Community-Acquired Pneumonia",
        "category": "clinical_guidelines",
    },
    {
        "content": """Stroke: Acute Evaluation and Management

Stroke is a medical emergency characterized by sudden onset of neurological deficits due to vascular causes.

Types:
1. Ischemic Stroke (87%): Thrombotic, Embolic, Lacunar
2. Hemorrhagic Stroke (13%): Intracerebral, Subarachnoid

Recognition (BE FAST):
B - Balance: Sudden loss of balance
E - Eyes: Sudden vision changes
F - Face: Facial droop
A - Arms: Arm weakness
S - Speech: Speech difficulty
T - Time: Time to call emergency services

Treatment:
Ischemic Stroke:
- IV thrombolysis (alteplase): Within 4.5 hours
- Mechanical thrombectomy: Within 6-24 hours for large vessel occlusion

Contraindications to Thrombolysis:
- Intracranial hemorrhage on CT
- Head trauma or stroke in past 3 months
- Major surgery in past 14 days
- Blood pressure > 185/110 despite treatment

Secondary Prevention:
- Antiplatelet therapy (aspirin, clopidogrel)
- Anticoagulation for cardioembolic stroke
- Statin therapy, BP control""",
        "source": "American Stroke Association",
        "title": "Guidelines for Early Management of Acute Ischemic Stroke",
        "category": "clinical_guidelines",
    },
    {
        "content": """Chronic Obstructive Pulmonary Disease (COPD)

COPD is a common, preventable, and treatable disease characterized by persistent respiratory symptoms and airflow limitation.

Pathophysiology:
- Chronic bronchitis: Chronic inflammation with mucus hypersecretion
- Emphysema: Destruction of alveolar walls

Risk Factors:
- Cigarette smoking (primary cause - 85-90%)
- Environmental exposures
- Alpha-1 antitrypsin deficiency

Clinical Presentation:
- Chronic cough, sputum production
- Dyspnea (initially on exertion, progressing to rest)
- Wheezing, chest tightness

Diagnosis:
Spirometry (Gold Standard): Post-bronchodilator FEV1/FVC < 0.70

GOLD Classification:
- GOLD 1 (Mild): FEV1 ≥ 80%
- GOLD 2 (Moderate): FEV1 50-79%
- GOLD 3 (Severe): FEV1 30-49%
- GOLD 4 (Very Severe): FEV1 < 30%

Management:
- Smoking cessation (most important)
- Bronchodilators (LABA, LAMA)
- Inhaled corticosteroids for exacerbations
- Pulmonary rehabilitation
- Oxygen therapy if PaO2 < 55 mmHg""",
        "source": "Global Initiative for Chronic Obstructive Lung Disease",
        "title": "GOLD Report: COPD Diagnosis, Management, and Prevention",
        "category": "disease_encyclopedia",
    },
    {
        "content": """Major Depressive Disorder: Clinical Overview and Treatment

Major depressive disorder (MDD) is a common but serious mood disorder characterized by persistent feelings of sadness and loss of interest.

Diagnostic Criteria (DSM-5):
Five or more symptoms for 2 weeks, including:
1. Depressed mood most of the day
2. Markedly diminished interest in activities
3. Significant weight change
4. Insomnia or hypersomnia
5. Psychomotor agitation or retardation
6. Fatigue or loss of energy
7. Feelings of worthlessness or excessive guilt
8. Diminished concentration
9. Recurrent thoughts of death

Treatment:
Psychotherapy:
- Cognitive Behavioral Therapy (CBT)
- Interpersonal Therapy (IPT)

Pharmacotherapy:
First-line SSRIs:
- Sertraline (50-200 mg)
- Escitalopram (10-20 mg)
- Fluoxetine (20-80 mg)

SNRIs:
- Venlafaxine (75-375 mg)
- Duloxetine (30-120 mg)

Treatment duration: 6-12 months after remission
Recurrence rate: 50% after one episode, 80% after two episodes""",
        "source": "American Psychiatric Association",
        "title": "Practice Guideline for Major Depressive Disorder",
        "category": "clinical_guidelines",
    },
    {
        "content": """Metformin: Comprehensive Drug Information

Drug Class: Biguanide antidiabetic agent

Mechanism of Action:
1. Decreases hepatic glucose production (primary)
2. Improves insulin sensitivity in peripheral tissues
3. Decreases intestinal glucose absorption

Indications:
- Type 2 diabetes mellitus (first-line therapy)
- Polycystic ovary syndrome (off-label)

Dosage:
Immediate-release: Start 500 mg once/twice daily, max 2550 mg/day
Extended-release: Start 500-1000 mg once daily, max 2000 mg/day

Contraindications:
- Severe renal impairment (eGFR < 30 mL/min)
- Acute or chronic metabolic acidosis
- Radiologic studies with iodinated contrast

Warnings:
Lactic Acidosis: Rare but serious; risk factors include renal impairment, liver disease, alcohol use

Adverse Effects:
Common: Diarrhea, nausea, abdominal pain, metallic taste
Less common: Vitamin B12 deficiency, lactic acidosis

Drug Interactions:
- Cimetidine: May increase metformin levels
- Alcohol: Increases lactic acidosis risk
- Iodinated contrast: Discontinue metformin before procedure

Patient Counseling:
- Take with meals to minimize GI upset
- Report unexplained muscle pain or weakness
- Limit alcohol consumption""",
        "source": "National Institutes of Health",
        "title": "Metformin Drug Information",
        "category": "drug_information",
    },
]


def load_sample_knowledge_base() -> MedicalKnowledgeBase:
    """Load the sample medical knowledge base"""
    kb = MedicalKnowledgeBase()

    for doc in SAMPLE_MEDICAL_KNOWLEDGE:
        kb.add_document(**doc)

    return kb


if __name__ == "__main__":
    # Test the document loader
    loader = MedicalDocumentLoader()
    kb = load_sample_knowledge_base()

    print(f"Knowledge Base Stats: {kb.get_stats()}")
    print(f"Total chunks: {len(kb.get_documents())}")
