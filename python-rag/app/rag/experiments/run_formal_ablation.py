"""Root entrypoint for the formal phase-1 ablation framework."""

from pathlib import Path
import sys


if __package__ in {None, ""}:
    # Allow direct file execution while keeping module execution as the documented path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.rag.experiments.phase1_formal_ablation import main


if __name__ == "__main__":
    main()
