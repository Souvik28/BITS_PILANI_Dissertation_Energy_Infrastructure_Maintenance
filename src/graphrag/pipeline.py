"""End-to-end orchestration.

Provides high-level functions the CLI scripts and Streamlit app call:

* ``generate_and_save``   — build synthetic dataset, graph, corpus, benchmark; persist.
* ``build_indexes``       — build + persist the vector index.
* ``load_system``         — reconstruct graph + vector store + engines from disk.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from . import config as cfgmod
from .benchmark import BenchmarkQuestion, build_benchmark
from .config import CONFIG, Config
from .data_generation import generate_dataset
from .embeddings import get_embedder
from .graph_store import BaseGraphStore, NetworkXGraphStore, get_graph_store
from .ingest import TextDoc, build_corpus, build_graph
from .llm import get_llm
from .rag import GraphRAG, NaiveVectorRAG
from .router import QueryRouter
from .vector_store import VectorStore


def generate_and_save(cfg: Config = CONFIG) -> dict:
    cfgmod.ensure_dirs()
    dataset = generate_dataset(cfg.datagen)
    with open(cfgmod.DATASET_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    corpus = build_corpus(dataset)
    with open(cfgmod.CORPUS_FILE, "w") as f:
        json.dump([d.to_dict() for d in corpus], f, indent=2)

    # Build graph just to derive the benchmark (always use an in-memory store here).
    graph = NetworkXGraphStore()
    build_graph(graph, dataset)
    benchmark = build_benchmark(graph, cfg.datagen, target=50)
    with open(cfgmod.BENCHMARK_FILE, "w") as f:
        json.dump([q.to_dict() for q in benchmark], f, indent=2)

    # Persist a backend-agnostic graph snapshot (nodes + edges) for reload.
    graph_snapshot = {
        "nodes": [n.to_dict() for n in graph.nodes()],
        "edges": [e.to_dict() for e in graph.edges()],
    }
    with open(cfgmod.GRAPH_FILE, "w") as f:
        json.dump(graph_snapshot, f, indent=2)

    return {
        "dataset": dataset["metadata"]["counts"],
        "corpus_docs": len(corpus),
        "benchmark_questions": len(benchmark),
    }


def load_dataset() -> dict:
    with open(cfgmod.DATASET_FILE) as f:
        return json.load(f)


def load_corpus() -> List[TextDoc]:
    with open(cfgmod.CORPUS_FILE) as f:
        return [TextDoc.from_dict(d) for d in json.load(f)]


def load_benchmark() -> List[BenchmarkQuestion]:
    with open(cfgmod.BENCHMARK_FILE) as f:
        return [BenchmarkQuestion.from_dict(d) for d in json.load(f)]


def _populate_graph(cfg: Config) -> BaseGraphStore:
    """Build a graph store from the persisted dataset, honouring the backend config."""
    dataset = load_dataset()
    store = get_graph_store(cfg.graph)
    # For a fresh Neo4j store, wipe before loading to stay idempotent.
    if hasattr(store, "wipe"):
        try:
            store.wipe()  # type: ignore[attr-defined]
        except Exception:
            pass
    build_graph(store, dataset)
    return store


def build_indexes(cfg: Config = CONFIG) -> dict:
    cfgmod.ensure_dirs()
    corpus = load_corpus()
    embedder = get_embedder(cfg.embedding)
    vs = VectorStore(embedder)
    vs.add_documents(corpus)
    vs.save(cfgmod.VECTOR_INDEX_FILE)

    # Also (re)load the graph into the configured backend (e.g. push to Neo4j).
    store = _populate_graph(cfg)
    node_count = len(store.nodes())
    store.close()

    return {
        "vector_docs": len(vs.docs),
        "embedding_dim": embedder.dim,
        "graph_nodes": node_count,
        "graph_backend": cfg.graph.backend,
    }


@dataclass
class RAGSystem:
    graph_store: BaseGraphStore
    vector_store: VectorStore
    naive: NaiveVectorRAG
    graphrag: GraphRAG
    router: QueryRouter
    cfg: Config

    def answer(self, question: str, mode: str = "auto"):
        if mode == "naive":
            return self.naive.answer(question)
        if mode == "graph":
            return self.graphrag.answer(question)
        if mode == "auto":
            decision = self.router.route(question)
            if decision.route == "vector":
                ans = self.naive.answer(question)
            else:
                ans = self.graphrag.answer(question)
            ans.mode = f"{ans.mode} (routed:{decision.route})"
            return ans
        return self.graphrag.answer(question)

    def close(self):
        self.graph_store.close()


def load_system(cfg: Config = CONFIG) -> RAGSystem:
    embedder = get_embedder(cfg.embedding)
    vs = VectorStore.load(cfgmod.VECTOR_INDEX_FILE, embedder)
    store = _populate_graph(cfg)
    llm = get_llm(cfg.llm)

    naive = NaiveVectorRAG(vs, llm, cfg)
    graphrag = GraphRAG(vs, store, llm, cfg)
    router = QueryRouter()

    return RAGSystem(
        graph_store=store,
        vector_store=vs,
        naive=naive,
        graphrag=graphrag,
        router=router,
        cfg=cfg,
    )
