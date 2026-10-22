"""Tests for the graph store + traversal primitives."""
from graphrag.schema import EdgeType, NodeLabel


def test_nodes_and_edges_loaded(graph, dataset):
    counts = dataset["metadata"]["counts"]
    total_nodes = (
        counts["assets"] + counts["components"] + counts["failure_modes"]
        + counts["sops"] + counts["work_orders"] + counts["telemetry"]
    )
    assert len(graph.nodes()) == total_nodes
    assert len(graph.edges()) > 0


def test_located_in_edges(graph):
    comps = graph.nodes_by_label(NodeLabel.COMPONENT)
    sample = comps[0]
    nbrs = graph.neighbors(sample.id, edge_types=[EdgeType.LOCATED_IN], direction="out")
    assert nbrs, "component should be LOCATED_IN an asset"
    _edge, asset = nbrs[0]
    assert asset.label == NodeLabel.ASSET


def test_downstream_traversal(graph):
    seps = [c for c in graph.nodes_by_label(NodeLabel.COMPONENT)
            if c.properties.get("type") == "Separator"]
    assert seps
    ds = graph.downstream(seps[0].id, max_depth=3)
    assert ds, "separator should feed downstream components"
    # downstream should not include the separator itself
    assert seps[0].id not in {n.id for n in ds}


def test_upstream_is_inverse_of_downstream(graph):
    seps = [c for c in graph.nodes_by_label(NodeLabel.COMPONENT)
            if c.properties.get("type") == "Separator"]
    sep = seps[0]
    ds = graph.downstream(sep.id, max_depth=1)
    assert ds
    child = ds[0]
    up = graph.upstream(child.id, max_depth=3)
    assert sep.id in {n.id for n in up}


def test_subgraph_respects_max_nodes(graph):
    seps = [c for c in graph.nodes_by_label(NodeLabel.COMPONENT)][:1]
    nodes, _edges = graph.subgraph([s.id for s in seps], hops=3, max_nodes=5)
    assert len(nodes) <= 5
