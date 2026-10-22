"""Step 2 — build the vector index and load the graph into the configured backend.

Usage:
    python -m scripts.build_index
"""
import scripts  # noqa: F401

from graphrag import config as cfgmod
from graphrag.config import CONFIG
from graphrag.pipeline import build_indexes


def main() -> None:
    print("Building indexes...")
    summary = build_indexes(CONFIG)
    print("\nIndex build complete:")
    for k, v in summary.items():
        print(f"  {k:>16}: {v}")
    print(f"\nVector index written to: {cfgmod.VECTOR_INDEX_FILE}")


if __name__ == "__main__":
    main()
