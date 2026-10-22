"""Step 4 — run the 50-question benchmark: Naive Vector RAG vs Graph-RAG.

Produces a comparison table + per-category breakdown and writes JSON results to
``data/results/``.

Usage:
    python -m scripts.run_evaluation
    python -m scripts.run_evaluation --limit 10     # quick smoke run
"""
import scripts  # noqa: F401

import argparse
import json
from datetime import datetime

from graphrag import config as cfgmod
from graphrag.config import CONFIG
from graphrag.evaluation import evaluate_system
from graphrag.pipeline import load_benchmark, load_system


def _fmt_row(name, agg):
    return (
        f"  {name:<22}"
        f"{agg.get('faithfulness', 0):>13.3f}"
        f"{agg.get('answer_relevance', 0):>12.3f}"
        f"{agg.get('context_precision', 0):>12.3f}"
        f"{agg.get('context_recall', 0):>11.3f}"
        f"{agg.get('answer_correctness', 0):>14.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions.")
    args = parser.parse_args()

    cfgmod.ensure_dirs()
    benchmark = load_benchmark()
    if args.limit:
        benchmark = benchmark[: args.limit]

    system = load_system(CONFIG)
    print(f"Evaluating {len(benchmark)} questions on two systems...")

    naive_pairs, graph_pairs = [], []
    try:
        for i, q in enumerate(benchmark, 1):
            naive_pairs.append((q, system.naive.answer(q.question)))
            graph_pairs.append((q, system.graphrag.answer(q.question)))
            if i % 10 == 0 or i == len(benchmark):
                print(f"  ...{i}/{len(benchmark)}")
    finally:
        system.close()

    naive_report = evaluate_system("Naive Vector RAG", naive_pairs)
    graph_report = evaluate_system("Graph-RAG", graph_pairs)

    header = (
        f"\n  {'System':<22}{'Faithfulness':>13}{'AnsRelev':>12}"
        f"{'CtxPrec':>12}{'CtxRecall':>11}{'Correctness':>14}"
    )
    print("\n" + "=" * 86)
    print("  RAG EVALUATION  (RAGAS-style metrics, higher is better)")
    print("=" * 86)
    print(header)
    print("  " + "-" * 82)
    print(_fmt_row("Naive Vector RAG", naive_report.aggregate()))
    print(_fmt_row("Graph-RAG", graph_report.aggregate()))

    # per-category correctness comparison
    print("\n  Answer correctness by category (naive -> graph-rag):")
    nc = naive_report.by_category()
    gc = graph_report.by_category()
    for cat in sorted(set(nc) | set(gc)):
        n = nc.get(cat, {}).get("answer_correctness", 0)
        g = gc.get(cat, {}).get("answer_correctness", 0)
        count = gc.get(cat, nc.get(cat, {})).get("n", 0)
        delta = g - n
        arrow = "UP" if delta > 0.001 else ("--" if abs(delta) <= 0.001 else "DN")
        print(f"    {cat:<12} (n={count:>2}): {n:.3f} -> {g:.3f}  [{arrow} {delta:+.3f}]")

    # persist
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": stamp,
        "config": {
            "graph_backend": CONFIG.graph.backend,
            "embed_backend": CONFIG.embedding.backend,
            "llm_backend": CONFIG.llm.backend,
        },
        "naive_vector": naive_report.to_dict(),
        "graph_rag": graph_report.to_dict(),
    }
    out_path = cfgmod.RESULTS_DIR / f"evaluation_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    latest = cfgmod.RESULTS_DIR / "evaluation_latest.json"
    with open(latest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
