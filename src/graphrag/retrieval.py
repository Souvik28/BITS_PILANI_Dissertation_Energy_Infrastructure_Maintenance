"""Retrieval engines: vector, graph, and hybrid.

* ``VectorRetriever`` — pure semantic search over the text corpus.
* ``GraphRetriever`` — links query entities to graph nodes, then extracts a topological
  subgraph (multi-hop, with explicit downstream/upstream handling of CONNECTED_TO).
* ``HybridRetriever`` — fuses both: vector hits seed graph traversal, and the subgraph is
  rendered back into context the LLM can ground against.

The ``RetrievalResult`` carries both the text passages and a structured description of the
subgraph so the grounding layer can verify topological claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .config import RetrievalConfig
from .graph_store import BaseGraphStore
from .schema import Edge, EdgeType, Node, NodeLabel
from .vector_store import SearchHit, VectorStore


@dataclass
class RetrievalResult:
    passages: List[str] = field(default_factory=list)
    vector_hits: List[SearchHit] = field(default_factory=list)
    subgraph_nodes: List[Node] = field(default_factory=list)
    subgraph_edges: List[Edge] = field(default_factory=list)
    linked_entities: List[Node] = field(default_factory=list)
    graph_facts: List[str] = field(default_factory=list)
    mode: str = "hybrid"

    def context_text(self) -> str:
        parts: List[str] = []
        if self.graph_facts:
            parts.append("STRUCTURED GRAPH FACTS:\n" + "\n".join(f"- {g}" for g in self.graph_facts))
        if self.passages:
            parts.append("RETRIEVED DOCUMENTS:\n" + "\n".join(f"- {p}" for p in self.passages))
        return "\n\n".join(parts)


# component-tag patterns: P1_SEP_V-202, P2_EQ_103, V-202, etc.
_TAG_RE = re.compile(
    r"\b(P\d+_(?:EQ|SEP)_[A-Z0-9\-]+|P\d+_\d{2,4}|[A-Z]{2,3}_(?:EQ|SEP)_[A-Z0-9\-]+|[VPKE]-\d{2,4})\b"
)
_TYPE_WORDS = {
    "pump": "Centrifugal_Pump",
    "pumps": "Centrifugal_Pump",
    "turbine": "Gas_Turbine",
    "turbines": "Gas_Turbine",
    "exchanger": "Heat_Exchanger",
    "exchangers": "Heat_Exchanger",
    "valve": "Pressure_Valve",
    "valves": "Pressure_Valve",
    "separator": "Separator",
    "separators": "Separator",
    "compressor": "Compressor",
    "compressors": "Compressor",
}


class EntityLinker:
    """Maps free-text query fragments to concrete graph node ids."""

    def __init__(self, store: BaseGraphStore):
        self.store = store
        self._by_name: Dict[str, Node] = {}
        self._by_id: Dict[str, Node] = {}
        for n in store.nodes():
            self._by_id[n.id.lower()] = n
            name = str(n.properties.get("name", "")).lower()
            if name:
                self._by_name[name] = n

    def link(self, query: str) -> List[Node]:
        found: Dict[str, Node] = {}
        q = query.strip()

        # 1. direct id / tag mentions
        for m in _TAG_RE.finditer(q):
            frag = m.group(1).replace(" ", "").replace("-", "").lower()
            for nid, node in self._by_id.items():
                norm = nid.replace("-", "").replace("_", "")
                if frag in norm or norm.endswith(frag):
                    found[node.id] = node

        # 2. name substring match (e.g. "Separator V-202")
        ql = q.lower()
        for name, node in self._by_name.items():
            if name and name in ql:
                found[node.id] = node

        return list(found.values())

    def detect_component_types(self, query: str) -> List[str]:
        ql = query.lower()
        types = []
        for word, ctype in _TYPE_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", ql) and ctype not in types:
                types.append(ctype)
        return types


class VectorRetriever:
    def __init__(self, vector_store: VectorStore, cfg: RetrievalConfig):
        self.vs = vector_store
        self.cfg = cfg

    def retrieve(self, query: str) -> RetrievalResult:
        hits = self.vs.search(query, top_k=self.cfg.top_k_vector)
        return RetrievalResult(
            passages=[h.doc.text for h in hits],
            vector_hits=hits,
            mode="vector",
        )


class GraphRetriever:
    def __init__(self, store: BaseGraphStore, cfg: RetrievalConfig):
        self.store = store
        self.cfg = cfg
        self.linker = EntityLinker(store)

    def _describe_node(self, n: Node) -> str:
        p = n.properties
        if n.label == NodeLabel.COMPONENT:
            return (f"{p.get('name', n.id)} [{n.id}] is a {p.get('type', '?').replace('_', ' ')} "
                    f"in {p.get('asset_id', '?')} (status {p.get('status', '?')}).")
        if n.label == NodeLabel.ASSET:
            return f"Asset {p.get('name', n.id)} [{n.id}] at {p.get('location', '?')}."
        if n.label == NodeLabel.FAILURE_MODE:
            return f"Failure mode {p.get('failure_code', n.id)} '{p.get('name', '')}': {p.get('symptom', '')}."
        if n.label == NodeLabel.SOP:
            return f"SOP {n.id} '{p.get('title', '')}' governs failure {p.get('failure_id', '?')}."
        if n.label == NodeLabel.WORK_ORDER:
            return f"Work order {n.id}: {p.get('summary', '')}"
        if n.label == NodeLabel.TELEMETRY:
            return (f"Telemetry {n.id} from {p.get('comp_id', '?')}: vibration "
                    f"{p.get('vibration_mm_s', '?')} mm/s, temp {p.get('temp_celsius', '?')} C.")
        return f"{n.label.value} {n.id}"

    def _edge_fact(self, e: Edge) -> str:
        verb = {
            EdgeType.CONNECTED_TO: "is connected to (feeds)",
            EdgeType.LOCATED_IN: "is located in",
            EdgeType.PERFORMED_ON: "was performed on",
            EdgeType.MEASURED_FROM: "was measured from",
            EdgeType.TRIGGERED_BY: "was triggered by",
            EdgeType.GOVERNS_REPAIR: "governs repair of",
        }[e.type]
        return f"{e.source} {verb} {e.target}."

    def retrieve(self, query: str) -> RetrievalResult:
        entities = self.linker.link(query)
        ql = query.lower()

        # Detect the component-type *filter* on the query's OWN target set, but ignore
        # type words that belong to an already-linked entity's proper name. Otherwise a
        # query like "downstream of Separator V-202" would wrongly filter the downstream
        # set to Separators only.
        query_for_types = ql
        for ent in entities:
            name = str(ent.properties.get("name", "")).lower()
            if name:
                query_for_types = query_for_types.replace(name, " ")
        type_filter = self.linker.detect_component_types(query_for_types)

        seed_ids: List[str] = [e.id for e in entities]
        extra_nodes: List[Node] = []
        graph_facts: List[str] = []
        directional_components: List[Node] = []

        # Directional reasoning: "downstream of X" / "upstream of X"
        for ent in entities:
            if ent.label != NodeLabel.COMPONENT:
                continue
            if "downstream" in ql:
                ds = self.store.downstream(ent.id, max_depth=self.cfg.graph_hops)
                for d in ds:
                    if not type_filter or d.properties.get("type") in type_filter:
                        extra_nodes.append(d)
                        directional_components.append(d)
                        seed_ids.append(d.id)
                        graph_facts.append(f"{d.properties.get('name', d.id)} is downstream of {ent.id}.")
            if "upstream" in ql:
                us = self.store.upstream(ent.id, max_depth=self.cfg.graph_hops)
                for u in us:
                    if not type_filter or u.properties.get("type") in type_filter:
                        extra_nodes.append(u)
                        directional_components.append(u)
                        seed_ids.append(u.id)
                        graph_facts.append(f"{u.properties.get('name', u.id)} is upstream of {ent.id}.")

        # Multi-hop: if the question is about failure modes, resolve the failure modes
        # applicable to the directionally-derived components. We combine two signals:
        #   (1) the component-type -> failure-mode ontology (schema-level truth), and
        #   (2) failure modes actually observed via Component <-PERFORMED_ON- WorkOrder
        #       -TRIGGERED_BY-> FailureMode history.
        # Both are multi-hop graph facts that naive vector retrieval cannot produce.
        if directional_components and "failure mode" in ql:
            from .data_generation import TYPE_TO_FAILURES

            fm_codes: Dict[str, Node] = {}
            for comp in directional_components:
                ctype = comp.properties.get("type")
                for code in TYPE_TO_FAILURES.get(ctype, []):
                    fm_node = self.store.get_node(code)
                    if fm_node:
                        fm_codes[code] = fm_node
                        seed_ids.append(code)
                for _e, wo in self.store.neighbors(
                    comp.id, edge_types=[EdgeType.PERFORMED_ON], direction="in"
                ):
                    for _e2, fm in self.store.neighbors(
                        wo.id, edge_types=[EdgeType.TRIGGERED_BY], direction="out"
                    ):
                        fm_codes[fm.id] = fm
                        seed_ids.append(fm.id)
            if fm_codes:
                codes_sorted = sorted(fm_codes)
                graph_facts.append(
                    "Applicable failure modes for the queried downstream/upstream "
                    f"components: {', '.join(codes_sorted)}."
                )
                for code, fm in fm_codes.items():
                    graph_facts.append(
                        f"Failure mode {code}: {fm.properties.get('name', '')} "
                        f"({fm.properties.get('symptom', '')})."
                    )

        # If no explicit entity but a component type is asked about, seed by type.
        if not seed_ids and type_filter:
            for n in self.store.nodes_by_label(NodeLabel.COMPONENT):
                if n.properties.get("type") in type_filter:
                    seed_ids.append(n.id)
            seed_ids = seed_ids[: self.cfg.max_graph_nodes]

        nodes, edges = self.store.subgraph(
            seed_ids, hops=self.cfg.graph_hops, max_nodes=self.cfg.max_graph_nodes
        )

        # include the directionally-derived nodes even if traversal capped them out
        node_index = {n.id: n for n in nodes}
        for n in extra_nodes + entities:
            node_index.setdefault(n.id, n)
        nodes = list(node_index.values())

        facts = graph_facts + [self._edge_fact(e) for e in edges]
        passages = [self._describe_node(n) for n in nodes]

        return RetrievalResult(
            passages=passages,
            subgraph_nodes=nodes,
            subgraph_edges=edges,
            linked_entities=entities,
            graph_facts=facts,
            mode="graph",
        )


class HybridRetriever:
    """Vector hits seed graph traversal; results are fused into one context."""

    def __init__(self, vector_store: VectorStore, store: BaseGraphStore, cfg: RetrievalConfig):
        self.vector = VectorRetriever(vector_store, cfg)
        self.graph = GraphRetriever(store, cfg)
        self.store = store
        self.cfg = cfg

    def retrieve(self, query: str) -> RetrievalResult:
        v = self.vector.retrieve(query)
        g = self.graph.retrieve(query)

        # Seed additional graph traversal from the node_ids of the top vector hits.
        vec_seed_ids: List[str] = []
        for hit in v.vector_hits[: self.cfg.top_k_vector]:
            vec_seed_ids.extend(hit.doc.node_ids)
        if vec_seed_ids:
            vnodes, vedges = self.store.subgraph(
                vec_seed_ids, hops=1, max_nodes=self.cfg.max_graph_nodes
            )
        else:
            vnodes, vedges = [], []

        # merge subgraphs
        node_index = {n.id: n for n in g.subgraph_nodes}
        for n in vnodes:
            node_index.setdefault(n.id, n)
        edge_index = {(e.source, e.target, e.type.value): e for e in g.subgraph_edges}
        for e in vedges:
            edge_index.setdefault((e.source, e.target, e.type.value), e)

        merged_nodes = list(node_index.values())
        merged_edges = list(edge_index.values())

        gr = self.graph  # for describe helpers
        facts = list(g.graph_facts)
        existing = set(facts)
        for e in merged_edges:
            f = gr._edge_fact(e)
            if f not in existing:
                facts.append(f)
                existing.add(f)

        return RetrievalResult(
            passages=v.passages + g.passages,
            vector_hits=v.vector_hits,
            subgraph_nodes=merged_nodes,
            subgraph_edges=merged_edges,
            linked_entities=g.linked_entities,
            graph_facts=facts,
            mode="hybrid",
        )
