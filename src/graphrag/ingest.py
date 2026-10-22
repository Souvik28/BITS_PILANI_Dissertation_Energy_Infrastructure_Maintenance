"""Ingestion: dataset dict -> knowledge graph + text corpus.

Builds the property graph (nodes + typed edges following the ontology) and, in parallel,
a list of text "documents" (one per SOP, work order, component, failure mode) that feeds
the vector index. Each document carries ``node_ids`` metadata so a vector hit can seed a
graph traversal — this linkage is what makes hybrid Graph-RAG possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .graph_store import BaseGraphStore
from .schema import Edge, EdgeType, Node, NodeLabel


@dataclass
class TextDoc:
    doc_id: str
    text: str
    source_type: str          # sop | work_order | component | failure_mode | asset
    node_ids: List[str] = field(default_factory=list)   # graph nodes this doc grounds to
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "source_type": self.source_type,
            "node_ids": self.node_ids,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict) -> "TextDoc":
        return TextDoc(
            doc_id=d["doc_id"],
            text=d["text"],
            source_type=d["source_type"],
            node_ids=d.get("node_ids", []),
            metadata=d.get("metadata", {}),
        )


def build_graph(store: BaseGraphStore, dataset: Dict) -> BaseGraphStore:
    """Populate the graph store from a generated dataset dict."""
    # Nodes
    for a in dataset["assets"]:
        store.add_node(Node(id=a["asset_id"], label=NodeLabel.ASSET, properties=a))
    for c in dataset["components"]:
        store.add_node(Node(id=c["comp_id"], label=NodeLabel.COMPONENT, properties=c))
    for f in dataset["failure_modes"]:
        store.add_node(Node(id=f["failure_id"], label=NodeLabel.FAILURE_MODE, properties=f))
    for s in dataset["sops"]:
        store.add_node(Node(id=s["sop_id"], label=NodeLabel.SOP, properties=s))
    for w in dataset["work_orders"]:
        store.add_node(Node(id=w["wo_id"], label=NodeLabel.WORK_ORDER, properties=w))
    for t in dataset["telemetry"]:
        store.add_node(Node(id=t["metric_id"], label=NodeLabel.TELEMETRY, properties=t))

    # Edges
    for c in dataset["components"]:
        store.add_edge(Edge(c["comp_id"], c["asset_id"], EdgeType.LOCATED_IN))
    for conn in dataset["connections"]:
        store.add_edge(Edge(conn["source"], conn["target"], EdgeType.CONNECTED_TO))
    for w in dataset["work_orders"]:
        store.add_edge(Edge(w["wo_id"], w["comp_id"], EdgeType.PERFORMED_ON))
        store.add_edge(Edge(w["wo_id"], w["failure_id"], EdgeType.TRIGGERED_BY))
    for t in dataset["telemetry"]:
        store.add_edge(Edge(t["metric_id"], t["comp_id"], EdgeType.MEASURED_FROM))
    for s in dataset["sops"]:
        store.add_edge(Edge(s["sop_id"], s["failure_id"], EdgeType.GOVERNS_REPAIR))

    return store


def build_corpus(dataset: Dict) -> List[TextDoc]:
    """Produce text documents for the vector index, each linked to graph node ids."""
    docs: List[TextDoc] = []

    for a in dataset["assets"]:
        docs.append(
            TextDoc(
                doc_id=f"doc_asset_{a['asset_id']}",
                text=(
                    f"Asset {a['name']} ({a['asset_id']}), system code {a['system_code']}, "
                    f"located at {a['location']}. Criticality tier {a['criticality_tier']}. "
                    f"Installed {a['installation_date']}."
                ),
                source_type="asset",
                node_ids=[a["asset_id"]],
                metadata={"name": a["name"]},
            )
        )

    for c in dataset["components"]:
        docs.append(
            TextDoc(
                doc_id=f"doc_comp_{c['comp_id']}",
                text=(
                    f"Component {c['name']} ({c['comp_id']}) is a {c['type'].replace('_', ' ')} "
                    f"located in asset {c['asset_id']}. Model {c['model']} by {c['manufacturer']}. "
                    f"Design pressure {c['design_pressure_bar']} bar, max operating temperature "
                    f"{c['operating_temp_max_c']} C. Current status: {c['status']}. "
                    f"Installed {c['installed_year']}."
                ),
                source_type="component",
                node_ids=[c["comp_id"], c["asset_id"]],
                metadata={"type": c["type"], "status": c["status"]},
            )
        )

    for f in dataset["failure_modes"]:
        docs.append(
            TextDoc(
                doc_id=f"doc_fm_{f['failure_id']}",
                text=(
                    f"Failure mode {f['failure_code']} '{f['name']}'. Symptom: {f['symptom']}. "
                    f"Root cause category: {f['root_cause_category']}. "
                    f"Severity index {f['severity_index']}/5."
                ),
                source_type="failure_mode",
                node_ids=[f["failure_id"]],
                metadata={"code": f["failure_code"]},
            )
        )

    for s in dataset["sops"]:
        docs.append(
            TextDoc(
                doc_id=f"doc_sop_{s['sop_id']}",
                text=f"{s['title']} ({s['sop_id']}, rev {s['revision']}). {s['body']}",
                source_type="sop",
                node_ids=[s["sop_id"], s["failure_id"]],
                metadata={"hazard_class": s["hazard_class"], "target": s["target_component_type"]},
            )
        )

    for w in dataset["work_orders"]:
        docs.append(
            TextDoc(
                doc_id=f"doc_wo_{w['wo_id']}",
                text=(
                    f"Work order {w['wo_id']} ({w['issue_type']}, {w['priority']}) created "
                    f"{w['creation_date']}. {w['summary']} Resolution: {w['resolution']} "
                    f"Cost ${w['cost_usd']}, {w['duration_hours']} hours, resolved by "
                    f"{w['resolved_by']}."
                ),
                source_type="work_order",
                node_ids=[w["wo_id"], w["comp_id"], w["failure_id"]],
                metadata={"priority": w["priority"], "issue_type": w["issue_type"]},
            )
        )

    return docs
