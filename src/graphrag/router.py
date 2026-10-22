"""Routing agent (LangGraph-style, rule-based).

Decides whether a query is best served by pure vector retrieval, graph traversal, or the
hybrid engine. The dissertation plan (Week 10) calls for a LangGraph routing agent; this
is a transparent, dependency-free implementation of the same decision policy. If LangGraph
is installed it can wrap this policy, but the policy itself is what matters for evaluation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Route = Literal["vector", "graph", "hybrid"]

# Signals that a query needs topology / multi-hop reasoning.
_GRAPH_SIGNALS = [
    r"\bdownstream\b", r"\bupstream\b", r"\bconnected\b", r"\bfeeds?\b",
    r"\bfeeding\b", r"\btopology\b", r"\bpath\b", r"\bbetween\b",
    r"\blocated in\b", r"\bwhich .* (connect|feed)", r"\bgoverns?\b",
]
# Signals that a query is a pure lookup / definition (vector is fine).
_VECTOR_SIGNALS = [
    r"\bwhat is\b", r"\bdefine\b", r"\bdescription\b", r"\bsymptom", r"\bmeaning of\b",
]


@dataclass
class RouteDecision:
    route: Route
    reason: str


class QueryRouter:
    def route(self, query: str) -> RouteDecision:
        q = query.lower()
        graph_hits = [p for p in _GRAPH_SIGNALS if re.search(p, q)]
        vector_hits = [p for p in _VECTOR_SIGNALS if re.search(p, q)]

        # topology words almost always imply graph reasoning
        if graph_hits:
            # if it also references documents (SOP/procedure), go hybrid
            if re.search(r"\bsop\b|\bprocedure\b|\bwork order\b|\bfailure mode", q):
                return RouteDecision("hybrid", f"Topology + document signals: {graph_hits}")
            return RouteDecision("hybrid", f"Topology signals present: {graph_hits}")

        if vector_hits and not graph_hits:
            return RouteDecision("vector", f"Lookup/definition signals: {vector_hits}")

        # default: hybrid is the safe, most-capable choice
        return RouteDecision("hybrid", "No strong signal; defaulting to hybrid retrieval.")
