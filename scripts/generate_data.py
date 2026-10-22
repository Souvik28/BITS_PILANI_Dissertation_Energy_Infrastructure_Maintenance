"""Step 1 — generate synthetic dataset, corpus, graph snapshot, and benchmark.

Usage:
    python -m scripts.generate_data
"""
import scripts  # noqa: F401  (sets up sys.path)

import json

from graphrag import config as cfgmod
from graphrag.config import CONFIG
from graphrag.pipeline import generate_and_save


def main() -> None:
    print("Generating synthetic energy-infrastructure dataset...")
    summary = generate_and_save(CONFIG)
    print("\nDataset counts:")
    for k, v in summary["dataset"].items():
        print(f"  {k:>14}: {v}")
    print(f"\n  corpus documents : {summary['corpus_docs']}")
    print(f"  benchmark Qs     : {summary['benchmark_questions']}")
    print(f"\nWritten to: {cfgmod.PROCESSED_DIR}")
    for p in (cfgmod.DATASET_FILE, cfgmod.CORPUS_FILE, cfgmod.GRAPH_FILE, cfgmod.BENCHMARK_FILE):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
