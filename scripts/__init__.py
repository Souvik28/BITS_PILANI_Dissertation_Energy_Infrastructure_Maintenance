"""CLI scripts for the Graph-RAG pipeline."""
import sys
from pathlib import Path

# Make ``src`` importable when running ``python -m scripts.xxx`` from the repo root.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
