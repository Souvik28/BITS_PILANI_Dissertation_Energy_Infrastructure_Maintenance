"""ISO 14224-adapted property-graph ontology for energy infrastructure.

Defines node labels, relationship types, and typed dataclasses that mirror the schema
in the dissertation plan (Part 1, Section 3). Keeping the ontology in one place lets the
ingestion, retrieval, and grounding layers share a single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeLabel(str, Enum):
    ASSET = "Asset"
    COMPONENT = "Component"
    WORK_ORDER = "WorkOrder"
    TELEMETRY = "Telemetry"
    SOP = "SOP"
    FAILURE_MODE = "FailureMode"


class EdgeType(str, Enum):
    CONNECTED_TO = "CONNECTED_TO"       # Component -> Component (process flow)
    LOCATED_IN = "LOCATED_IN"           # Component -> Asset
    PERFORMED_ON = "PERFORMED_ON"       # WorkOrder -> Component
    MEASURED_FROM = "MEASURED_FROM"     # Telemetry -> Component
    TRIGGERED_BY = "TRIGGERED_BY"       # WorkOrder -> FailureMode
    GOVERNS_REPAIR = "GOVERNS_REPAIR"   # SOP -> FailureMode


# Which node label owns which primary-id field, for generic handling.
PRIMARY_ID = {
    NodeLabel.ASSET: "asset_id",
    NodeLabel.COMPONENT: "comp_id",
    NodeLabel.WORK_ORDER: "wo_id",
    NodeLabel.TELEMETRY: "metric_id",
    NodeLabel.SOP: "sop_id",
    NodeLabel.FAILURE_MODE: "failure_id",
}


@dataclass
class Node:
    """A generic property-graph node."""

    id: str
    label: NodeLabel
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "label": self.label.value, "properties": self.properties}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Node":
        return Node(id=d["id"], label=NodeLabel(d["label"]), properties=d.get("properties", {}))


@dataclass
class Edge:
    """A directed, typed relationship."""

    source: str
    target: str
    type: EdgeType
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "properties": self.properties,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Edge":
        return Edge(
            source=d["source"],
            target=d["target"],
            type=EdgeType(d["type"]),
            properties=d.get("properties", {}),
        )


# --------------------------------------------------------------------------- domain types
@dataclass
class Asset:
    asset_id: str
    name: str
    system_code: str
    criticality_tier: int
    installation_date: str
    location: str

    def as_props(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Component:
    comp_id: str
    asset_id: str
    name: str
    type: str
    model: str
    manufacturer: str
    design_pressure_bar: float
    operating_temp_max_c: float
    installed_year: int
    status: str

    def as_props(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FailureMode:
    failure_id: str
    failure_code: str
    name: str
    symptom: str
    root_cause_category: str
    severity_index: int

    def as_props(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkOrder:
    wo_id: str
    comp_id: str
    failure_id: str
    creation_date: str
    issue_type: str
    priority: str
    cost_usd: float
    duration_hours: float
    resolved_by: str
    summary: str
    resolution: str

    def as_props(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Telemetry:
    metric_id: str
    comp_id: str
    timestamp: str
    vibration_mm_s: float
    temp_celsius: float
    pressure_bar: float
    flow_rate_m3h: float

    def as_props(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SOP:
    sop_id: str
    failure_id: str
    title: str
    revision: str
    target_component_type: str
    hazard_class: str
    approved_year: int
    body: str

    def as_props(self) -> Dict[str, Any]:
        return asdict(self)


CRITICALITY_TIERS = {1: "Critical", 2: "High", 3: "Medium", 4: "Low"}

COMPONENT_TYPES = [
    "Centrifugal_Pump",
    "Gas_Turbine",
    "Heat_Exchanger",
    "Pressure_Valve",
    "Separator",
    "Compressor",
]
