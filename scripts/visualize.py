"""Step 2.5 — render the data-schema visualizations to PNG files.

Produces five figures covering both structured and unstructured data and writes them to
``data/figures/``:

    01_ontology.png             the property-graph meta-schema (node labels + edges)
    02_knowledge_graph.png      the actual knowledge graph, colored by node label
    03_asset_topology.png       the CONNECTED_TO process flow for one platform
    04_schema_distributions.png attribute distributions across the dataset
    05_text_overview.png        unstructured text corpus overview (doc length + terms)

Usage:
    python -m scripts.visualize
    python -m scripts.visualize --asset PLATFORM_ALPHA_2   # topology for a chosen platform

Prerequisite: run `python -m scripts.generate_data` first. Needs matplotlib + networkx
(`pip install matplotlib networkx`).
"""
import scripts  # noqa: F401

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe for CLI

from graphrag import config as cfgmod
from graphrag.config import CONFIG
from graphrag.graph_store import get_graph_store
from graphrag.ingest import build_graph
from graphrag import pipeline
from graphrag import visualization as viz


def main() -> None:
    parser = argparse.ArgumentParser(description="Render data-schema visualizations.")
    parser.add_argument("--asset", default=None, help="Asset id for the topology plot.")
    parser.add_argument("--include-telemetry", action="store_true",
                        help="Include Telemetry nodes in the knowledge-graph plot.")
    args = parser.parse_args()

    cfgmod.ensure_dirs()
    try:
        dataset = pipeline.load_dataset()
        corpus = pipeline.load_corpus()
    except FileNotFoundError:
        raise SystemExit("Dataset not found. Run `python -m scripts.generate_data` first.")

    store = get_graph_store(CONFIG.graph)
    build_graph(store, dataset)

    import matplotlib.pyplot as plt

    jobs = {
        "01_ontology.png": lambda: viz.plot_ontology(),
        "02_knowledge_graph.png": lambda: viz.plot_knowledge_graph(
            store, exclude_labels=[] if args.include_telemetry else None
        ),
        "03_asset_topology.png": lambda: viz.plot_asset_topology(store, asset_id=args.asset),
        "04_schema_distributions.png": lambda: viz.plot_schema_distributions(dataset),
        "05_text_overview.png": lambda: viz.plot_text_overview(corpus),
    }

    print(f"Rendering {len(jobs)} figures to {cfgmod.FIGURES_DIR} ...")
    for name, make in jobs.items():
        fig = make()
        path = cfgmod.FIGURES_DIR / name
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  - {name}")

    store.close()
    print("\nDone. Open the PNGs in data/figures/ or view them in the notebook/Streamlit app.")


if __name__ == "__main__":
    main()
