"""Step 3 — ask a maintenance question against the Graph-RAG system.

Usage:
    python -m scripts.run_query "What are the failure modes for pumps downstream of Separator V-202?"
    python -m scripts.run_query --mode naive "..."     # baseline
    python -m scripts.run_query --mode graph "..."     # hybrid graph-rag
    python -m scripts.run_query --compare "..."        # run both, side by side
"""
import scripts  # noqa: F401

import argparse

from graphrag.config import CONFIG
from graphrag.pipeline import load_system
from graphrag.router import QueryRouter


def _print_answer(title, ans):
    print(f"\n{'=' * 78}\n{title}  [mode={ans.mode}]\n{'=' * 78}")
    print(ans.answer)
    if ans.grounding:
        print(f"\n  groundedness: {ans.grounding.score}")
        if ans.grounding.unsupported_claims:
            print(f"  unsupported : {len(ans.grounding.unsupported_claims)} claim(s)")
    if ans.graph_facts:
        print(f"  graph facts : {len(ans.graph_facts)} topological facts used")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Graph-RAG maintenance assistant.")
    parser.add_argument("question", help="The maintenance question to ask.")
    parser.add_argument("--mode", choices=["auto", "naive", "graph"], default="auto")
    parser.add_argument("--compare", action="store_true", help="Run naive vs graph-rag.")
    args = parser.parse_args()

    system = load_system(CONFIG)
    try:
        if args.compare:
            print(f"\nQUESTION: {args.question}")
            route = QueryRouter().route(args.question)
            print(f"ROUTER  : {route.route} -- {route.reason}")
            _print_answer("NAIVE VECTOR RAG (baseline)", system.answer(args.question, mode="naive"))
            _print_answer("GRAPH-RAG (hybrid)", system.answer(args.question, mode="graph"))
        else:
            ans = system.answer(args.question, mode=args.mode)
            _print_answer(f"ANSWER", ans)
    finally:
        system.close()


if __name__ == "__main__":
    main()
