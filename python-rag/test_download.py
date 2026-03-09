import sys
print("Python executable:", sys.executable, file=sys.stderr)
print("Python version:", sys.version, file=sys.stderr)
print("Starting test...", file=sys.stderr)

try:
    print("Before import datasets")
    from datasets import load_dataset
    print("Successfully imported datasets")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

print("Test completed successfully")
