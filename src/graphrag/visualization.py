"""Visualization helpers for the generated data schema.

Covers BOTH sides of the dataset:

* **Structured** — the property graph: ontology/meta-schema, the knowledge graph colored
  by node label, per-platform CONNECTED_TO process topology, and attribute distributions
  (component types, failure modes, work-order priorities, telemetry).
* **Unstructured** — the text corpus (SOP bodies, work-order summaries): document-length
  distribution per source type and a term-frequency view (text-as-data).

Rendering uses matplotlib (static, notebook-friendly). Every function returns the
matplotlib ``Figure`` so a notebook can display it inline and a script can save it. A
NetworkX graph is used for layout; a deterministic seed keeps figures reproducible.

Import is lazy/guarded so importing this module never breaks the core pipeline if
matplotlib is not installed — the functions raise a clear error only when actually called.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .schema import EdgeType, NodeLabel

# Consistent color per node label across every figure.
LABEL_COLORS: Dict[str, str] = {
    NodeLabel.ASSET.value: "#1f77b4",        # blue
    NodeLabel.COMPONENT.value: "#2ca02c",    # green
    NodeLabel.WORK_ORDER.value: "#ff7f0e",   # orange
    NodeLabel.TELEMETRY.value: "#9467bd",    # purple
    NodeLabel.SOP.value: "#8c564b",          # brown
    NodeLabel.FAILURE_MODE.value: "#d62728", # red
}

EDGE_COLORS: Dict[str, str] = {
    EdgeType.CONNECTED_TO.value: "#2ca02c",
    EdgeType.LOCATED_IN.value: "#1f77b4",
    EdgeType.PERFORMED_ON.value: "#ff7f0e",
    EdgeType.MEASURED_FROM.value: "#9467bd",
    EdgeType.TRIGGERED_BY.value: "#d62728",
    EdgeType.GOVERNS_REPAIR.value: "#8c564b",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "are", "was", "from", "has", "have",
    "been", "under", "then", "its", "into", "per", "via", "not", "but", "all", "any",
    "can", "will", "which", "applied", "reported", "tested", "within", "requires",
    "require", "standard", "operating", "procedure", "failure", "mode", "component",
    "components", "work", "order", "based", "retrieved", "knowledge", "documents",
}


def _require_mpl():
    try:
        import matplotlib
        import matplotlib.pyplot as plt  # noqa: F401
        return matplotlib
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for visualization. Install it with "
            "`pip install matplotlib` (or `pip install -r requirements.txt`)."
        ) from exc


def _require_nx():
    try:
        import networkx as nx
        return nx
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "networkx is required for graph layout. Install it with `pip install networkx`."
        ) from exc


# --------------------------------------------------------------------------- ontology
def plot_ontology(figsize: Tuple[int, int] = (10, 7)):
    """Draw the META-schema: the 6 node labels and the 6 relationship types.

    This is the 'data schema types' view — the ontology itself, independent of any data.
    """
    _require_mpl()
    nx = _require_nx()
    import matplotlib.pyplot as plt

    g = nx.DiGraph()
    for label in NodeLabel:
        g.add_node(label.value)

    schema_edges = [
        (NodeLabel.COMPONENT.value, NodeLabel.COMPONENT.value, EdgeType.CONNECTED_TO.value),
        (NodeLabel.COMPONENT.value, NodeLabel.ASSET.value, EdgeType.LOCATED_IN.value),
        (NodeLabel.WORK_ORDER.value, NodeLabel.COMPONENT.value, EdgeType.PERFORMED_ON.value),
        (NodeLabel.TELEMETRY.value, NodeLabel.COMPONENT.value, EdgeType.MEASURED_FROM.value),
        (NodeLabel.WORK_ORDER.value, NodeLabel.FAILURE_MODE.value, EdgeType.TRIGGERED_BY.value),
        (NodeLabel.SOP.value, NodeLabel.FAILURE_MODE.value, EdgeType.GOVERNS_REPAIR.value),
    ]
    for s, t, rel in schema_edges:
        g.add_edge(s, t, label=rel)

    pos = {
        NodeLabel.ASSET.value: (0.0, 0.0),
        NodeLabel.COMPONENT.value: (1.5, 0.0),
        NodeLabel.TELEMETRY.value: (1.5, 1.4),
        NodeLabel.WORK_ORDER.value: (3.0, 0.7),
        NodeLabel.FAILURE_MODE.value: (4.5, 0.0),
        NodeLabel.SOP.value: (4.5, 1.4),
    }

    fig, ax = plt.subplots(figsize=figsize)
    node_colors = [LABEL_COLORS[n] for n in g.nodes()]
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=3200,
                           edgecolors="black", ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=9, font_color="white",
                            font_weight="bold", ax=ax)

    # self-loop (CONNECTED_TO) + directed edges
    for s, t, d in g.edges(data=True):
        rel = d["label"]
        color = EDGE_COLORS.get(rel, "#555555")
        if s == t:
            ax.annotate("", xy=(pos[s][0] + 0.25, pos[s][1] + 0.35),
                        xytext=(pos[s][0] - 0.25, pos[s][1] + 0.35),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        connectionstyle="arc3,rad=1.2"))
            ax.text(pos[s][0], pos[s][1] + 0.7, rel, ha="center", fontsize=8, color=color)
        else:
            nx.draw_networkx_edges(g, pos, edgelist=[(s, t)], edge_color=color,
                                   width=2, arrowsize=22, ax=ax,
                                   connectionstyle="arc3,rad=0.08")
            mx, my = (pos[s][0] + pos[t][0]) / 2, (pos[s][1] + pos[t][1]) / 2
            ax.text(mx, my + 0.12, rel, ha="center", fontsize=7.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    ax.set_title("Property-Graph Ontology (ISO 14224-adapted)\nNode labels and relationship types",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.margins(0.15)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- knowledge graph
def _build_nx_from_store(store):
    nx = _require_nx()
    g = nx.DiGraph()
    for n in store.nodes():
        g.add_node(n.id, label=n.label.value, properties=n.properties)
    for e in store.edges():
        g.add_edge(e.source, e.target, type=e.type.value)
    return g


def plot_knowledge_graph(
    store,
    exclude_labels: Optional[List[str]] = None,
    max_nodes: int = 250,
    figsize: Tuple[int, int] = (14, 11),
    seed: int = 42,
):
    """Draw the ACTUAL knowledge graph, nodes colored by label.

    Telemetry is excluded by default (hundreds of leaf nodes add clutter); pass
    ``exclude_labels=[]`` to include everything.
    """
    _require_mpl()
    nx = _require_nx()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    exclude = set(exclude_labels if exclude_labels is not None else [NodeLabel.TELEMETRY.value])
    g_full = _build_nx_from_store(store)

    keep = [n for n, d in g_full.nodes(data=True) if d["label"] not in exclude]
    g = g_full.subgraph(keep).copy()

    if g.number_of_nodes() > max_nodes:
        # keep highest-degree nodes to stay legible
        top = sorted(g.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        g = g.subgraph([n for n, _ in top]).copy()

    pos = nx.spring_layout(g, seed=seed, k=0.6, iterations=80)

    fig, ax = plt.subplots(figsize=figsize)
    # edges by type
    for etype, color in EDGE_COLORS.items():
        elist = [(u, v) for u, v, d in g.edges(data=True) if d.get("type") == etype]
        if elist:
            nx.draw_networkx_edges(g, pos, edgelist=elist, edge_color=color,
                                   width=0.8, alpha=0.5, arrowsize=8, ax=ax)
    # nodes by label
    for label, color in LABEL_COLORS.items():
        nlist = [n for n, d in g.nodes(data=True) if d["label"] == label]
        if nlist:
            sizes = [120 + 40 * g.degree(n) for n in nlist]
            nx.draw_networkx_nodes(g, pos, nodelist=nlist, node_color=color,
                                   node_size=sizes, alpha=0.9, edgecolors="black",
                                   linewidths=0.3, ax=ax)

    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                     markersize=10, label=l)
              for l, c in LABEL_COLORS.items() if l not in exclude]
    ax.legend(handles=legend, loc="upper right", fontsize=9, title="Node label")
    ax.set_title(f"Knowledge Graph — {g.number_of_nodes()} nodes, {g.number_of_edges()} edges"
                 + (f" (excluding {', '.join(sorted(exclude))})" if exclude else ""),
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_asset_topology(
    store,
    asset_id: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    seed: int = 7,
):
    """Draw the CONNECTED_TO process-flow topology for one platform (structured view).

    Highlights the separator root and labels components by short name, so the
    upstream/downstream flow the dissertation reasons over is visible.
    """
    _require_mpl()
    nx = _require_nx()
    import matplotlib.pyplot as plt

    comps = store.nodes_by_label(NodeLabel.COMPONENT)
    if asset_id is None:
        # pick the platform that has a separator with the most downstream reach
        seps = [c for c in comps if c.properties.get("type") == "Separator"]
        seps.sort(key=lambda c: len(store.downstream(c.id, max_depth=4)), reverse=True)
        asset_id = seps[0].properties.get("asset_id") if seps else comps[0].properties.get("asset_id")

    members = [c for c in comps if c.properties.get("asset_id") == asset_id]
    member_ids = {c.id for c in members}

    g = nx.DiGraph()
    for c in members:
        g.add_node(c.id, name=c.properties.get("name", c.id),
                   type=c.properties.get("type", ""),
                   is_sep=c.properties.get("type") == "Separator")
    for e in store.edges():
        if e.type == EdgeType.CONNECTED_TO and e.source in member_ids and e.target in member_ids:
            g.add_edge(e.source, e.target)

    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        pos = nx.spring_layout(g, seed=seed, k=1.0, iterations=100)

    fig, ax = plt.subplots(figsize=figsize)
    sep_nodes = [n for n, d in g.nodes(data=True) if d["is_sep"]]
    other_nodes = [n for n, d in g.nodes(data=True) if not d["is_sep"]]

    nx.draw_networkx_edges(g, pos, edge_color="#2ca02c", width=1.6,
                           arrowsize=16, ax=ax, connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_nodes(g, pos, nodelist=sep_nodes, node_color="#d62728",
                           node_size=1600, node_shape="s", edgecolors="black", ax=ax,
                           label="Separator (flow root)")
    nx.draw_networkx_nodes(g, pos, nodelist=other_nodes, node_color="#2ca02c",
                           node_size=900, edgecolors="black", ax=ax, label="Component")
    labels = {n: g.nodes[n]["name"].replace("Separator ", "").replace("Centrifugal ", "")
              for n in g.nodes()}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=7, ax=ax)

    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"Process-Flow Topology (CONNECTED_TO) — {asset_id}",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- distributions
def plot_schema_distributions(dataset: Dict, figsize: Tuple[int, int] = (14, 10)):
    """Bar/pie panels summarising the structured attributes across the dataset."""
    _require_mpl()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 1. node-label counts
    counts = dataset["metadata"]["counts"]
    label_map = {
        "assets": NodeLabel.ASSET.value, "components": NodeLabel.COMPONENT.value,
        "work_orders": NodeLabel.WORK_ORDER.value, "telemetry": NodeLabel.TELEMETRY.value,
        "sops": NodeLabel.SOP.value, "failure_modes": NodeLabel.FAILURE_MODE.value,
    }
    labels = [k for k in label_map if k in counts]
    vals = [counts[k] for k in labels]
    colors = [LABEL_COLORS[label_map[k]] for k in labels]
    axes[0, 0].bar(labels, vals, color=colors)
    axes[0, 0].set_title("Node counts by label")
    axes[0, 0].tick_params(axis="x", rotation=40)
    for i, v in enumerate(vals):
        axes[0, 0].text(i, v, str(v), ha="center", va="bottom", fontsize=8)

    # 2. component types
    ctypes = Counter(c["type"] for c in dataset["components"])
    axes[0, 1].barh(list(ctypes.keys()), list(ctypes.values()), color="#2ca02c")
    axes[0, 1].set_title("Component types")

    # 3. component status (pie)
    status = Counter(c["status"] for c in dataset["components"])
    axes[0, 2].pie(status.values(), labels=status.keys(), autopct="%1.0f%%",
                   colors=["#2ca02c", "#ff7f0e", "#d62728"])
    axes[0, 2].set_title("Component status")

    # 4. failure-mode usage in work orders
    fm_counts = Counter(w["failure_id"] for w in dataset["work_orders"])
    fm_name = {f["failure_id"]: f["failure_code"] for f in dataset["failure_modes"]}
    keys = sorted(fm_counts)
    axes[1, 0].bar([fm_name.get(k, k) for k in keys], [fm_counts[k] for k in keys],
                   color="#d62728")
    axes[1, 0].set_title("Work orders per failure mode")
    axes[1, 0].tick_params(axis="x", rotation=40)

    # 5. work-order priority
    prio = Counter(w["priority"] for w in dataset["work_orders"])
    pk = sorted(prio)
    axes[1, 1].bar(pk, [prio[k] for k in pk], color="#ff7f0e")
    axes[1, 1].set_title("Work-order priority")
    axes[1, 1].tick_params(axis="x", rotation=30)

    # 6. telemetry vibration distribution
    vib = [t["vibration_mm_s"] for t in dataset["telemetry"]]
    axes[1, 2].hist(vib, bins=30, color="#9467bd", edgecolor="white")
    axes[1, 2].axvline(4.5, color="red", linestyle="--", label="4.5 mm/s alarm")
    axes[1, 2].set_title("Telemetry: vibration (mm/s)")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("Structured Data — Schema Attribute Distributions",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# --------------------------------------------------------------------------- unstructured text
def plot_text_overview(corpus, figsize: Tuple[int, int] = (14, 6)):
    """Visualise the UNSTRUCTURED text corpus: doc-length by source type + top terms.

    ``corpus`` is a list of ``TextDoc`` (or dicts with 'text'/'source_type').
    """
    _require_mpl()
    import matplotlib.pyplot as plt

    def _text(d):
        return d.text if hasattr(d, "text") else d["text"]

    def _stype(d):
        return d.source_type if hasattr(d, "source_type") else d["source_type"]

    by_type: Dict[str, List[int]] = {}
    all_tokens: Counter = Counter()
    for d in corpus:
        txt = _text(d)
        by_type.setdefault(_stype(d), []).append(len(txt.split()))
        for tok in _TOKEN_RE.findall(txt.lower()):
            if tok not in _STOPWORDS:
                all_tokens[tok] += 1

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. doc-length distribution per source type (box plot)
    types = sorted(by_type)
    box_data = [by_type[t] for t in types]
    # 'labels' was renamed 'tick_labels' in matplotlib 3.9; support both.
    try:
        axes[0].boxplot(box_data, tick_labels=types, patch_artist=True)
    except TypeError:
        axes[0].boxplot(box_data, labels=types, patch_artist=True)
    axes[0].set_title("Document length (words) by source type")
    axes[0].set_ylabel("words per document")
    axes[0].tick_params(axis="x", rotation=30)

    # 2. top terms across the corpus
    top = all_tokens.most_common(20)
    terms = [t for t, _ in top][::-1]
    freqs = [c for _, c in top][::-1]
    axes[1].barh(terms, freqs, color="#8c564b")
    axes[1].set_title("Top 20 terms in unstructured text (stopwords removed)")

    fig.suptitle("Unstructured Data — Text Corpus Overview",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --------------------------------------------------------------------------- export helpers
def save_all_figures(store, dataset, corpus, out_dir, dpi: int = 130) -> List[str]:
    """Render every figure and save to ``out_dir``. Returns the list of written paths."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    figs = {
        "01_ontology.png": plot_ontology(),
        "02_knowledge_graph.png": plot_knowledge_graph(store),
        "03_asset_topology.png": plot_asset_topology(store),
        "04_schema_distributions.png": plot_schema_distributions(dataset),
        "05_text_overview.png": plot_text_overview(corpus),
    }
    for name, fig in figs.items():
        path = out / name
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written
