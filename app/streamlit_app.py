"""Interactive maintenance decision-support dashboard (Week 10 / Week 15 deliverable).

Run with:  streamlit run app/streamlit_app.py

Lets a maintenance engineer:
  * ask a question and see the routed answer,
  * compare Naive Vector RAG vs Graph-RAG side by side,
  * inspect the retrieved subgraph, the graph facts used, and the groundedness score.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from graphrag import pipeline  # noqa: E402
from graphrag.config import CONFIG  # noqa: E402
from graphrag.graph_store import get_graph_store  # noqa: E402
from graphrag.ingest import build_graph  # noqa: E402
from graphrag.pipeline import load_system  # noqa: E402
from graphrag.router import QueryRouter  # noqa: E402

st.set_page_config(page_title="Graph-RAG Maintenance Assistant", layout="wide")


@st.cache_resource(show_spinner="Loading Graph-RAG system...")
def _get_system():
    return load_system(CONFIG)


@st.cache_resource(show_spinner="Loading dataset & graph for visualization...")
def _get_data_and_graph():
    """Load the dataset, corpus, and an in-memory graph for the visualization tab."""
    dataset = pipeline.load_dataset()
    corpus = pipeline.load_corpus()
    store = get_graph_store(CONFIG.graph)
    build_graph(store, dataset)
    return dataset, corpus, store


EXAMPLES = [
    "What are the failure modes for pumps connected downstream of Separator V-202?",
    "Which components are directly connected downstream of Separator V-202?",
    "Which SOP governs the repair of failure mode FM-01?",
    "What is the symptom of failure mode FM-02?",
    "In which asset is a degraded compressor located?",
]


def main():
    st.title("Graph-RAG — Grounded Decision Support for Energy Maintenance")
    st.caption(
        "BITS Pilani dissertation demo · Shell-context energy infrastructure · "
        f"graph={CONFIG.graph.backend} · embeddings={CONFIG.embedding.backend} · llm={CONFIG.llm.backend}"
    )

    ask_tab, viz_tab = st.tabs(["Ask", "Data schema"])
    with ask_tab:
        _ask_tab()
    with viz_tab:
        _viz_tab()


def _ask_tab():
    try:
        system = _get_system()
    except FileNotFoundError:
        st.error(
            "Indexes not found. Run `python -m scripts.generate_data` then "
            "`python -m scripts.build_index` first."
        )
        return

    with st.sidebar:
        st.header("Settings")
        mode = st.radio(
            "Answer mode",
            ["auto (router)", "graph-rag", "naive vector", "compare"],
            index=0,
        )
        st.markdown("---")
        st.subheader("Example questions")
        for ex in EXAMPLES:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state["question"] = ex

    question = st.text_input(
        "Ask a maintenance question",
        value=st.session_state.get("question", EXAMPLES[0]),
    )

    if st.button("Run", type="primary") or question:
        route = QueryRouter().route(question)
        st.info(f"**Router decision:** `{route.route}` — {route.reason}")

        if mode == "compare":
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Naive Vector RAG")
                _render(system.answer(question, mode="naive"))
            with c2:
                st.subheader("Graph-RAG (hybrid)")
                _render(system.answer(question, mode="graph"))
        else:
            mmap = {"auto (router)": "auto", "graph-rag": "graph", "naive vector": "naive"}
            _render(system.answer(question, mode=mmap[mode]))


def _viz_tab():
    """Visualise the generated data schema — structured graph + unstructured text."""
    st.subheader("Generated data schema — structured & unstructured")
    st.caption(
        "Explore the property-graph ontology, the knowledge graph, process-flow topology, "
        "attribute distributions, and the unstructured text corpus."
    )

    try:
        from graphrag import visualization as viz
    except Exception as exc:  # pragma: no cover
        st.error(f"Visualization needs matplotlib + networkx: {exc}")
        return

    try:
        dataset, corpus, store = _get_data_and_graph()
    except FileNotFoundError:
        st.error("Dataset not found. Run `python -m scripts.generate_data` first.")
        return

    counts = dataset["metadata"]["counts"]
    cols = st.columns(len(counts))
    for col, (k, v) in zip(cols, counts.items()):
        col.metric(k.replace("_", " ").title(), v)

    view = st.selectbox(
        "Choose a view",
        [
            "1 · Ontology (meta-schema)",
            "2 · Knowledge graph",
            "3 · Asset process-flow topology",
            "4 · Structured attribute distributions",
            "5 · Unstructured text overview",
        ],
    )

    if view.startswith("1"):
        st.pyplot(viz.plot_ontology())
    elif view.startswith("2"):
        include_telem = st.checkbox("Include Telemetry nodes (adds clutter)", value=False)
        excl = [] if include_telem else None
        with st.spinner("Laying out the graph..."):
            st.pyplot(viz.plot_knowledge_graph(store, exclude_labels=excl))
    elif view.startswith("3"):
        from graphrag.schema import NodeLabel
        assets = sorted({c.properties.get("asset_id")
                         for c in store.nodes_by_label(NodeLabel.COMPONENT)})
        asset = st.selectbox("Platform", assets)
        st.pyplot(viz.plot_asset_topology(store, asset_id=asset))
    elif view.startswith("4"):
        st.pyplot(viz.plot_schema_distributions(dataset))
    else:
        st.pyplot(viz.plot_text_overview(corpus))
        st.markdown("**Sample unstructured documents**")
        shown = {"sop": False, "work_order": False}
        for d in corpus:
            if d.source_type in shown and not shown[d.source_type]:
                st.text_area(f"{d.source_type} · {d.doc_id}", d.text, height=120)
                shown[d.source_type] = True
            if all(shown.values()):
                break


def _render(ans):
    st.markdown(f"**Answer** _({ans.mode})_")
    st.write(ans.answer)

    if ans.grounding:
        score = ans.grounding.score
        st.metric("Groundedness", f"{score:.2f}")
        st.progress(min(1.0, score))
        if ans.grounding.unsupported_claims:
            with st.expander(f"Unsupported claims ({len(ans.grounding.unsupported_claims)})"):
                for c in ans.grounding.unsupported_claims:
                    st.write(f"- {c}")

    if ans.graph_facts:
        with st.expander(f"Graph facts used ({len(ans.graph_facts)})"):
            for f in ans.graph_facts[:40]:
                st.write(f"- {f}")

    if ans.retrieval and ans.retrieval.subgraph_nodes:
        with st.expander(f"Retrieved subgraph ({len(ans.retrieval.subgraph_nodes)} nodes)"):
            for n in ans.retrieval.subgraph_nodes[:40]:
                st.write(f"- `{n.id}` ({n.label.value})")

    with st.expander(f"Retrieved passages ({len(ans.contexts)})"):
        for p in ans.contexts[:12]:
            st.write(f"- {p}")


if __name__ == "__main__":
    main()
