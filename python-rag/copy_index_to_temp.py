"""Copy FAISS index to temp directory without Chinese characters"""
import shutil
from pathlib import Path
import tempfile

# Source path
source_dir = Path(__file__).parent / "data" / "vector_store" / "faiss_index"

# Create temp directory
temp_dir = Path(tempfile.gettempdir()) / "medical_rag_faiss"
temp_dir.mkdir(exist_ok=True)

# Copy files
files_to_copy = ["index.faiss", "index.pkl", "metadata.json", "build_metadata.json"]
for filename in files_to_copy:
    src = source_dir / filename
    dst = temp_dir / filename
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied {filename} to {temp_dir}")
    else:
        print(f"Warning: {filename} not found")

print(f"\nIndex copied to: {temp_dir}")
print(f"Update your config to use: {temp_dir}")
