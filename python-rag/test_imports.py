import sys
import os

print("Testing imports...")

try:
    from app.rag.document_loader import Document

    print("  [OK] document_loader imports successful")
except Exception as e:
    print(f"  [FAIL] document_loader import failed: {e}")

try:
    from app.rag import (
        create_rag_pipeline,
        RAGConfig,
        MedicalRAGSystem,
        RuleBasedEvaluator,
    )

    print("  [OK] app.rag imports successful")
except Exception as e:
    print(f"  [FAIL] app.rag import failed: {e}")

try:
    from app.main import app

    print("  [OK] app.main imports successful")
except Exception as e:
    print(f"  [FAIL] app.main import failed: {e}")

print("\nAll imports completed!")
