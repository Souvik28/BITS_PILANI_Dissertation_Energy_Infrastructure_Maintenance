"""Ground-truth benchmark generator (Week 4 deliverable).

Derives ~50 evaluation questions directly from the generated graph so each has a
deterministic, verifiable gold answer and a set of gold evidence node ids. Questions span
four difficulty classes:

* ``lookup``       — single-node fact (favours vector).
* ``topology``     — downstream/upstream / connectivity (favours graph).
* ``multi_hop``    — failure modes of components downstream of X (favours Graph-RAG).
* ``procedural``   — which SOP governs the repair for a failure on component X (hybrid).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from .config import DataGenConfig
from .graph_store import BaseGraphStore
from .schema import EdgeType, NodeLabel
from .data_generation import TYPE_TO_FAILURES


@dataclass
class BenchmarkQuestion:
    qid: str
    question: str
    category: str
    gold_answer: str
    gold_node_ids: List[str] = field(default_factory=list)
    gold_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "qid": self.qid,
            "question": self.question,
            "category": self.category,
            "gold_answer": self.gold_answer,
            "gold_node_ids": self.gold_node_ids,
            "gold_keywords": self.gold_keywords,
        }

    @staticmethod
    def from_dict(d: Dict) -> "BenchmarkQuestion":
        return BenchmarkQuestion(**d)


def _downstream_components(store: BaseGraphStore, comp_id: str, type_filter=None):
    nodes = store.downstream(comp_id, max_depth=3)
    if type_filter:
        nodes = [n for n in nodes if n.properties.get("type") == type_filter]
    return nodes


def build_benchmark(store: BaseGraphStore, cfg: DataGenConfig, target: int = 50) -> List[BenchmarkQuestion]:
    rng = random.Random(cfg.seed + 7)
    questions: List[BenchmarkQuestion] = []

    components = store.nodes_by_label(NodeLabel.COMPONENT)
    assets = store.nodes_by_label(NodeLabel.ASSET)
    failure_modes = store.nodes_by_label(NodeLabel.FAILURE_MODE)
    sops = store.nodes_by_label(NodeLabel.SOP)
    separators = [c for c in components if c.properties.get("type") == "Separator"]

    sop_by_failure = {s.properties.get("failure_id"): s for s in sops}

    qn = 0

    def nid():
        nonlocal qn
        qn += 1
        return f"Q{qn:03d}"

    # ---- lookup questions (component facts)
    for comp in rng.sample(components, min(10, len(components))):
        p = comp.properties
        questions.append(
            BenchmarkQuestion(
                qid=nid(),
                question=f"What is the operating status and type of component {comp.id}?",
                category="lookup",
                gold_answer=(
                    f"{p.get('name')} ({comp.id}) is a {p.get('type', '').replace('_', ' ')} "
                    f"with status {p.get('status')}."
                ),
                gold_node_ids=[comp.id],
                gold_keywords=[comp.id, p.get("type", ""), p.get("status", "")],
            )
        )

    # ---- lookup questions (failure mode symptoms)
    for fm in failure_modes:
        p = fm.properties
        questions.append(
            BenchmarkQuestion(
                qid=nid(),
                question=f"What is the symptom of failure mode {p.get('failure_code')}?",
                category="lookup",
                gold_answer=f"Failure mode {p.get('failure_code')} '{p.get('name')}' presents: {p.get('symptom')}.",
                gold_node_ids=[fm.id],
                gold_keywords=[p.get("failure_code", ""), p.get("symptom", "")],
            )
        )

    # ---- topology: what does a separator feed?
    for sep in separators:
        ds = store.downstream(sep.id, max_depth=1)
        if not ds:
            continue
        names = [d.properties.get("name", d.id) for d in ds]
        questions.append(
            BenchmarkQuestion(
                qid=nid(),
                question=f"Which components are directly connected downstream of {sep.properties.get('name')}?",
                category="topology",
                gold_answer=f"{sep.properties.get('name')} directly feeds: {', '.join(names)}.",
                gold_node_ids=[sep.id] + [d.id for d in ds],
                gold_keywords=[d.id for d in ds],
            )
        )

    # ---- topology: where is component X located?
    for comp in rng.sample(components, min(6, len(components))):
        p = comp.properties
        questions.append(
            BenchmarkQuestion(
                qid=nid(),
                question=f"In which asset is component {comp.id} located?",
                category="topology",
                gold_answer=f"{comp.id} is located in asset {p.get('asset_id')}.",
                gold_node_ids=[comp.id, p.get("asset_id")],
                gold_keywords=[p.get("asset_id", "")],
            )
        )

    # ---- multi-hop: failure modes for <type> downstream of a separator
    for sep in separators:
        for ctype in ["Centrifugal_Pump", "Heat_Exchanger"]:
            ds = _downstream_components(store, sep.id, type_filter=ctype)
            if not ds:
                continue
            codes = sorted({c for c in TYPE_TO_FAILURES.get(ctype, [])})
            questions.append(
                BenchmarkQuestion(
                    qid=nid(),
                    question=(
                        f"What are the failure modes for {ctype.replace('_', ' ').lower()}s "
                        f"connected downstream of {sep.properties.get('name')}?"
                    ),
                    category="multi_hop",
                    gold_answer=(
                        f"{ctype.replace('_', ' ')}s downstream of {sep.properties.get('name')} "
                        f"are subject to failure modes {', '.join(codes)}."
                    ),
                    # gold_node_ids = all evidence the retriever should recover (the
                    # separator, the downstream components, and the failure modes).
                    gold_node_ids=[sep.id] + [d.id for d in ds] + codes,
                    # gold_keywords = the ANSWER content being assessed: the failure codes.
                    # The specific component ids are evidence, scored by context_recall.
                    gold_keywords=codes,
                )
            )

    # ---- procedural: which SOP governs the failure on component X
    for comp in rng.sample(components, min(12, len(components))):
        ctype = comp.properties.get("type")
        codes = TYPE_TO_FAILURES.get(ctype, [])
        if not codes:
            continue
        code = codes[0]
        sop = sop_by_failure.get(code)
        if not sop:
            continue
        questions.append(
            BenchmarkQuestion(
                qid=nid(),
                question=(
                    f"Which SOP governs the repair of failure mode {code}, which can affect "
                    f"component {comp.id}?"
                ),
                category="procedural",
                gold_answer=(
                    f"{sop.id} ('{sop.properties.get('title')}') governs repair of failure "
                    f"mode {code}, applicable to {comp.id}."
                ),
                gold_node_ids=[comp.id, code, sop.id],
                gold_keywords=[sop.id, code],
            )
        )

    # Top-up with additional component lookups if we are short of the target, so the
    # benchmark reliably reaches ``target`` questions regardless of graph size.
    if len(questions) < target:
        remaining = [c for c in components]
        rng.shuffle(remaining)
        seen_lookup = {q.gold_node_ids[0] for q in questions
                       if q.category == "lookup" and q.gold_node_ids}
        for comp in remaining:
            if len(questions) >= target:
                break
            if comp.id in seen_lookup:
                continue
            p = comp.properties
            questions.append(
                BenchmarkQuestion(
                    qid=nid(),
                    question=f"Which manufacturer and model is component {comp.id}?",
                    category="lookup",
                    gold_answer=(
                        f"{comp.id} is model {p.get('model')} manufactured by {p.get('manufacturer')}."
                    ),
                    gold_node_ids=[comp.id],
                    gold_keywords=[comp.id, p.get("manufacturer", ""), p.get("model", "")],
                )
            )

    rng.shuffle(questions)
    questions = questions[:target]
    # renumber for stable ids
    for i, q in enumerate(questions, 1):
        q.qid = f"Q{i:03d}"
    return questions
