"""Grounding / constraint-checking layer.

Verifies that an answer's claims are supported by the retrieved subgraph and passages.
This is the dissertation's core contribution: recommendations must explicitly match the
upstream/downstream equipment configuration, not just sound plausible.

Produces a ``GroundingReport`` with a 0-1 groundedness score and the list of claims that
could or could not be anchored to retrieved evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

from .retrieval import RetrievalResult

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]+")


def _tokens(text: str) -> Set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


@dataclass
class GroundingReport:
    score: float
    supported_claims: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return self.score >= 0.6


class GroundingChecker:
    def __init__(self, min_overlap: float = 0.34):
        self.min_overlap = min_overlap

    def check(self, answer: str, retrieval: RetrievalResult) -> GroundingReport:
        evidence_text = " ".join(retrieval.passages + retrieval.graph_facts)
        evidence_tokens = _tokens(evidence_text)

        # ids referenced in evidence, used for citation
        evidence_ids = sorted(
            {n.id for n in retrieval.subgraph_nodes}
            | {h.doc.doc_id for h in retrieval.vector_hits}
        )

        supported: List[str] = []
        unsupported: List[str] = []

        sentences = [s.strip() for s in _SENT_SPLIT.split(answer) if s.strip()]
        for sent in sentences:
            stoks = _tokens(sent)
            if not stoks:
                continue
            # content tokens only (drop very common words)
            content = {t for t in stoks if len(t) > 3}
            if not content:
                supported.append(sent)
                continue
            overlap = len(content & evidence_tokens) / len(content)
            if overlap >= self.min_overlap:
                supported.append(sent)
            else:
                unsupported.append(sent)

        total = len(supported) + len(unsupported)
        score = (len(supported) / total) if total else 0.0
        return GroundingReport(
            score=round(score, 3),
            supported_claims=supported,
            unsupported_claims=unsupported,
            evidence_ids=evidence_ids,
        )
