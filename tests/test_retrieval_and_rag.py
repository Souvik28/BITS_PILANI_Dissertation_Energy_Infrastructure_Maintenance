"""Tests for embeddings, vector store, retrieval, routing, grounding, and RAG."""
from graphrag.config import CONFIG, EmbeddingConfig
from graphrag.embeddings import HashingEmbedder, get_embedder
from graphrag.grounding import GroundingChecker
from graphrag.llm import ExtractiveLLM
from graphrag.rag import GraphRAG, NaiveVectorRAG
from graphrag.retrieval import EntityLinker, GraphRetriever, HybridRetriever, VectorRetriever
from graphrag.router import QueryRouter
from graphrag.vector_store import VectorStore


def _vector_store(corpus):
    emb = HashingEmbedder(dim=256)
    vs = VectorStore(emb)
    vs.add_documents(corpus)
    return vs


def test_hashing_embedder_normalised():
    emb = HashingEmbedder(dim=64)
    v = emb.embed_one("centrifugal pump bearing vibration")
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_vector_search_returns_relevant(corpus):
    vs = _vector_store(corpus)
    hits = vs.search("bearing seal degradation high vibration", top_k=5)
    assert hits
    joined = " ".join(h.doc.text.lower() for h in hits)
    assert "vibration" in joined or "bearing" in joined


def test_entity_linker_finds_separator(graph):
    linker = EntityLinker(graph)
    # find a separator tag from the graph
    from graphrag.schema import NodeLabel
    sep = [c for c in graph.nodes_by_label(NodeLabel.COMPONENT)
           if c.properties.get("type") == "Separator"][0]
    tag = sep.properties["name"]  # e.g. "Separator V-202"
    linked = linker.link(f"failure modes downstream of {tag}")
    assert sep.id in {n.id for n in linked}


def test_graph_retriever_downstream(graph):
    from graphrag.schema import NodeLabel
    # pick a separator that actually feeds at least one component
    seps = [c for c in graph.nodes_by_label(NodeLabel.COMPONENT)
            if c.properties.get("type") == "Separator"]
    sep = next(s for s in seps if graph.downstream(s.id, max_depth=2))
    tag = sep.properties["name"]
    retr = GraphRetriever(graph, CONFIG.retrieval)
    # type-agnostic query so the result does not depend on which equipment the
    # random generator placed downstream of this particular separator.
    result = retr.retrieve(f"What is connected downstream of {tag}?")
    assert result.subgraph_nodes
    assert any("downstream" in f for f in result.graph_facts), result.graph_facts[:5]


def test_router_routes_topology_to_hybrid():
    r = QueryRouter()
    assert r.route("components downstream of Separator V-202").route == "hybrid"
    assert r.route("what is failure mode FM-01").route in ("vector", "hybrid")


def test_grounding_detects_unsupported():
    from graphrag.retrieval import RetrievalResult
    res = RetrievalResult(passages=["Pump EQ_101 is operational in PLATFORM_ALPHA_1."],
                          graph_facts=["EQ_101 is located in PLATFORM_ALPHA_1."])
    checker = GroundingChecker()
    good = checker.check("Pump EQ_101 is operational in PLATFORM_ALPHA_1.", res)
    bad = checker.check("The reactor exploded due to alien interference yesterday.", res)
    assert good.score > bad.score


def test_naive_and_graph_rag_answer(graph, corpus):
    vs = _vector_store(corpus)
    llm = ExtractiveLLM()
    naive = NaiveVectorRAG(vs, llm, CONFIG)
    graphrag = GraphRAG(vs, graph, llm, CONFIG)

    q = "What is the symptom of failure mode FM-01?"
    a1 = naive.answer(q)
    a2 = graphrag.answer(q)
    assert a1.answer and a2.answer
    assert a1.mode == "naive_vector"
    assert a2.mode == "graph_rag"
    assert a2.grounding is not None
