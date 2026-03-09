#!/usr/bin/env python
import sys
import os

os.chdir(r"F:\课程\Semester B\CS6520 Project\RAG_Medical_Final\python-rag")
sys.path.insert(0, ".")

print("=" * 50)
print("Testing Medical RAG System")
print("=" * 50)

try:
    print("\n1. Importing modules...")
    from app.rag.api_medical_rag import MedicalRAGSystem, MedicalRAGConfig

    print("   ✓ Modules imported")

    print("\n2. Creating RAG system with custom config...")
    config = MedicalRAGConfig(
        llm_provider="deepseek",
        llm_model="2656053fa69c4c2d89c5a691d9d737c3",
        llm_api_key="6fcecb364d0647d2883e7f1d3f19d5b9",
        llm_base_url="https://wishub-x6.ctyun.cn/v1",
    )
    print("   ✓ Config created")

    print("\n3. Initializing system...")
    system = MedicalRAGSystem(config)
    system.initialize()
    print(f"   ✓ System initialized with {len(system.documents)} documents")

    print("\n4. Testing query...")
    result = system.query("高血压的一线治疗方案是什么？")
    print(f"   ✓ Query processed")
    print(f"\nAnswer: {result['answer'][:200]}...")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback

    traceback.print_exc()
