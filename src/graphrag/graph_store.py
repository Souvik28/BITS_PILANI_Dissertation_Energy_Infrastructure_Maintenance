"""Graph backend abstraction.

Two implementations behind one interface:

* ``NetworkXGraphStore`` — default, zero external services. If NetworkX itself is not
  installed it transparently uses a tiny pure-python adjacency implementation.
* ``Neo4jGraphStore``   — production backend; used when ``GRAPHRAG_GRAPH=neo4j`` and the
  ``neo4j`` driver + a reachable database are available.

Both expose the same traversal primitives the retrieval layer needs: neighbour lookup,
multi-hop subgraph extraction, directional (downstream/upstream) traversal of
CONNECTED_TO edges, and label/property lookups.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .config import GraphConfig
from .schema import Edge, EdgeType, Node, NodeLabel


class BaseGraphStore:
    """Interface shared by all graph backends."""

    def add_node(self, node: Node) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def add_edge(self, edge: Edge) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def get_node(self, node_id: str) -> Optional[Node]:  # pragma: no cover
        raise NotImplementedError

    def nodes(self) -> List[Node]:  # pragma: no cover - interface
        raise NotImplementedError

    def edges(self) -> List[Edge]:  # pragma: no cover - interface
        raise NotImplementedError

    def neighbors(
        self,
        node_id: str,
        edge_types: Optional[Iterable[EdgeType]] = None,
        direction: str = "both",
    ) -> List[Tuple[Edge, Node]]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass

    # ---------------------------------------------------- shared derived queries
    def subgraph(
        self,
        seed_ids: Iterable[str],
        hops: int = 2,
        max_nodes: int = 40,
    ) -> Tuple[List[Node], List[Edge]]:
        """BFS expansion from seed nodes up to ``hops``; capped at ``max_nodes``."""
        seen: Set[str] = set()
        out_nodes: List[Node] = []
        out_edges: List[Edge] = []
        frontier: deque = deque()

        for sid in seed_ids:
            n = self.get_node(sid)
            if n and sid not in seen:
                seen.add(sid)
                out_nodes.append(n)
                frontier.append((sid, 0))

        while frontier and len(out_nodes) < max_nodes:
            node_id, depth = frontier.popleft()
            if depth >= hops:
                continue
            for edge, neighbor in self.neighbors(node_id, direction="both"):
                out_edges.append(edge)
                if neighbor.id not in seen:
                    seen.add(neighbor.id)
                    out_nodes.append(neighbor)
                    frontier.append((neighbor.id, depth + 1))
                    if len(out_nodes) >= max_nodes:
                        break

        # de-dup edges
        uniq: Dict[Tuple[str, str, str], Edge] = {}
        for e in out_edges:
            uniq[(e.source, e.target, e.type.value)] = e
        return out_nodes, list(uniq.values())

    def downstream(self, comp_id: str, max_depth: int = 5) -> List[Node]:
        """Follow CONNECTED_TO edges in the flow direction (source -> target)."""
        return self._directional(comp_id, max_depth, forward=True)

    def upstream(self, comp_id: str, max_depth: int = 5) -> List[Node]:
        """Follow CONNECTED_TO edges against the flow direction (target <- source)."""
        return self._directional(comp_id, max_depth, forward=False)

    def _directional(self, comp_id: str, max_depth: int, forward: bool) -> List[Node]:
        seen = {comp_id}
        result: List[Node] = []
        frontier = deque([(comp_id, 0)])
        direction = "out" if forward else "in"
        while frontier:
            nid, depth = frontier.popleft()
            if depth >= max_depth:
                continue
            for _edge, neighbor in self.neighbors(
                nid, edge_types=[EdgeType.CONNECTED_TO], direction=direction
            ):
                if neighbor.id not in seen:
                    seen.add(neighbor.id)
                    result.append(neighbor)
                    frontier.append((neighbor.id, depth + 1))
        return result

    def nodes_by_label(self, label: NodeLabel) -> List[Node]:
        return [n for n in self.nodes() if n.label == label]


# --------------------------------------------------------------------------- NetworkX
class NetworkXGraphStore(BaseGraphStore):
    """Default backend. Uses networkx if available, else a pure-python fallback."""

    def __init__(self) -> None:
        try:
            import networkx as nx  # type: ignore

            self._nx = nx
            self._g = nx.MultiDiGraph()
            self._use_nx = True
        except Exception:
            self._nx = None
            self._use_nx = False
            self._node_data: Dict[str, Node] = {}
            self._out: Dict[str, List[Edge]] = defaultdict(list)
            self._in: Dict[str, List[Edge]] = defaultdict(list)

    # ---- mutation
    def add_node(self, node: Node) -> None:
        if self._use_nx:
            self._g.add_node(node.id, label=node.label.value, properties=node.properties)
        else:
            self._node_data[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        if self._use_nx:
            self._g.add_edge(
                edge.source,
                edge.target,
                key=edge.type.value,
                type=edge.type.value,
                properties=edge.properties,
            )
        else:
            self._out[edge.source].append(edge)
            self._in[edge.target].append(edge)

    # ---- lookups
    def get_node(self, node_id: str) -> Optional[Node]:
        if self._use_nx:
            if node_id not in self._g:
                return None
            d = self._g.nodes[node_id]
            return Node(id=node_id, label=NodeLabel(d["label"]), properties=d.get("properties", {}))
        return self._node_data.get(node_id)

    def nodes(self) -> List[Node]:
        if self._use_nx:
            return [
                Node(id=n, label=NodeLabel(d["label"]), properties=d.get("properties", {}))
                for n, d in self._g.nodes(data=True)
            ]
        return list(self._node_data.values())

    def edges(self) -> List[Edge]:
        if self._use_nx:
            out = []
            for u, v, k, d in self._g.edges(keys=True, data=True):
                out.append(
                    Edge(source=u, target=v, type=EdgeType(d["type"]),
                         properties=d.get("properties", {}))
                )
            return out
        seen: List[Edge] = []
        for lst in self._out.values():
            seen.extend(lst)
        return seen

    def neighbors(
        self,
        node_id: str,
        edge_types: Optional[Iterable[EdgeType]] = None,
        direction: str = "both",
    ) -> List[Tuple[Edge, Node]]:
        type_filter = {t.value for t in edge_types} if edge_types else None
        results: List[Tuple[Edge, Node]] = []

        if self._use_nx:
            if node_id not in self._g:
                return []
            if direction in ("out", "both"):
                for _u, v, k, d in self._g.out_edges(node_id, keys=True, data=True):
                    if type_filter and d["type"] not in type_filter:
                        continue
                    nb = self.get_node(v)
                    if nb:
                        results.append((Edge(node_id, v, EdgeType(d["type"]), d.get("properties", {})), nb))
            if direction in ("in", "both"):
                for u, _v, k, d in self._g.in_edges(node_id, keys=True, data=True):
                    if type_filter and d["type"] not in type_filter:
                        continue
                    nb = self.get_node(u)
                    if nb:
                        results.append((Edge(u, node_id, EdgeType(d["type"]), d.get("properties", {})), nb))
            return results

        # pure-python
        if direction in ("out", "both"):
            for e in self._out.get(node_id, []):
                if type_filter and e.type.value not in type_filter:
                    continue
                nb = self.get_node(e.target)
                if nb:
                    results.append((e, nb))
        if direction in ("in", "both"):
            for e in self._in.get(node_id, []):
                if type_filter and e.type.value not in type_filter:
                    continue
                nb = self.get_node(e.source)
                if nb:
                    results.append((e, nb))
        return results


# --------------------------------------------------------------------------- Neo4j
class Neo4jGraphStore(BaseGraphStore):
    """Neo4j-backed store. Requires the official ``neo4j`` python driver."""

    def __init__(self, cfg: GraphConfig) -> None:
        from neo4j import GraphDatabase  # type: ignore

        self.cfg = cfg
        self._driver = GraphDatabase.driver(
            cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)
        )
        # fail fast if the DB is unreachable
        self._driver.verify_connectivity()
        self._db = cfg.neo4j_database

    def _run(self, query: str, **params):
        with self._driver.session(database=self._db) as session:
            return list(session.run(query, **params))

    def wipe(self) -> None:
        self._run("MATCH (n) DETACH DELETE n")

    def add_node(self, node: Node) -> None:
        self._run(
            f"MERGE (n:{node.label.value} {{id: $id}}) SET n += $props",
            id=node.id,
            props=node.properties,
        )

    def add_edge(self, edge: Edge) -> None:
        self._run(
            f"MATCH (a {{id: $src}}), (b {{id: $dst}}) "
            f"MERGE (a)-[r:{edge.type.value}]->(b) SET r += $props",
            src=edge.source,
            dst=edge.target,
            props=edge.properties,
        )

    def get_node(self, node_id: str) -> Optional[Node]:
        rows = self._run("MATCH (n {id: $id}) RETURN labels(n) AS labels, properties(n) AS props LIMIT 1", id=node_id)
        if not rows:
            return None
        r = rows[0]
        labels = [l for l in r["labels"] if l in {nl.value for nl in NodeLabel}]
        label = NodeLabel(labels[0]) if labels else NodeLabel.COMPONENT
        props = dict(r["props"])
        props.pop("id", None)
        return Node(id=node_id, label=label, properties=props)

    def nodes(self) -> List[Node]:
        rows = self._run("MATCH (n) RETURN n.id AS id, labels(n) AS labels, properties(n) AS props")
        out = []
        valid = {nl.value for nl in NodeLabel}
        for r in rows:
            labels = [l for l in r["labels"] if l in valid]
            label = NodeLabel(labels[0]) if labels else NodeLabel.COMPONENT
            props = dict(r["props"])
            props.pop("id", None)
            out.append(Node(id=r["id"], label=label, properties=props))
        return out

    def edges(self) -> List[Edge]:
        rows = self._run(
            "MATCH (a)-[r]->(b) RETURN a.id AS src, b.id AS dst, type(r) AS t, properties(r) AS props"
        )
        valid = {et.value for et in EdgeType}
        return [
            Edge(r["src"], r["dst"], EdgeType(r["t"]), dict(r["props"]))
            for r in rows
            if r["t"] in valid
        ]

    def neighbors(
        self,
        node_id: str,
        edge_types: Optional[Iterable[EdgeType]] = None,
        direction: str = "both",
    ) -> List[Tuple[Edge, Node]]:
        type_clause = ""
        if edge_types:
            type_clause = ":" + "|".join(t.value for t in edge_types)
        if direction == "out":
            pattern = f"(n {{id: $id}})-[r{type_clause}]->(m)"
        elif direction == "in":
            pattern = f"(n {{id: $id}})<-[r{type_clause}]-(m)"
        else:
            pattern = f"(n {{id: $id}})-[r{type_clause}]-(m)"
        rows = self._run(
            f"MATCH {pattern} RETURN type(r) AS t, startNode(r).id AS src, endNode(r).id AS dst, "
            f"properties(r) AS rprops, m.id AS mid, labels(m) AS mlabels, properties(m) AS mprops",
            id=node_id,
        )
        valid_n = {nl.value for nl in NodeLabel}
        results: List[Tuple[Edge, Node]] = []
        for r in rows:
            labels = [l for l in r["mlabels"] if l in valid_n]
            label = NodeLabel(labels[0]) if labels else NodeLabel.COMPONENT
            mprops = dict(r["mprops"])
            mprops.pop("id", None)
            node = Node(id=r["mid"], label=label, properties=mprops)
            edge = Edge(r["src"], r["dst"], EdgeType(r["t"]), dict(r["rprops"]))
            results.append((edge, node))
        return results

    def close(self) -> None:
        self._driver.close()


# --------------------------------------------------------------------------- factory
def get_graph_store(cfg: GraphConfig) -> BaseGraphStore:
    """Return a graph store per config, falling back to NetworkX if Neo4j is unavailable."""
    if cfg.backend == "neo4j":
        try:
            store = Neo4jGraphStore(cfg)
            return store
        except Exception as exc:  # driver missing or DB unreachable
            if not cfg.allow_fallback:
                raise
            print(f"[graph_store] Neo4j unavailable ({exc!s}); falling back to NetworkX.")
    return NetworkXGraphStore()
