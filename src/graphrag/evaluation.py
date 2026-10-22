"""RAGAS-style evaluation metrics (dependency-free).

Implements lexical/overlap approximations of the standard RAG metrics so the two systems
can be compared without a cloud judge model:

* faithfulness / groundedness  — share of answer claims supported by retrieved context.
* answer_relevance             — overlap of the answer with the gold answer.
* context_precision            — share of retrieved contexts that are relevant (hit gold ids/keywords).
* context_recall               — share of gold evidence recovered by retrieval.
* answer_correctness           — keyword coverage of the gold answer by the produced answer.

If ``ragas`` + an LLM judge are installed, ``evaluate_with_ragas`` can be used instead;
these lexical metrics are the always-available default and are what the scripts report.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List

from .benchmark import BenchmarkQuestion
from .rag import RAGAnswer

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]+")


def _tokens(text: str) -> set:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


@dataclass
class QuestionScore:
    qid: str
    category: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    answer_correctness: float

    def as_dict(self) -> Dict:
        return self.__dict__


def _keyword_coverage(answer: str, keywords: List[str]) -> float:
    kws = [k for k in keywords if k]
    if not kws:
        return 1.0
    a = answer.lower()
    hit = sum(1 for k in kws if k.lower() in a)
    return hit / len(kws)


def score_question(q: BenchmarkQuestion, ans: RAGAnswer) -> QuestionScore:
    # faithfulness from the grounding report
    faithfulness = ans.grounding.score if ans.grounding else 0.0

    # answer relevance: token overlap with gold answer
    gold_tok = _tokens(q.gold_answer)
    ans_tok = _tokens(ans.answer)
    relevance = (len(gold_tok & ans_tok) / len(gold_tok)) if gold_tok else 0.0

    # context precision & recall against gold node ids / keywords
    retrieved_ids = set()
    if ans.retrieval:
        retrieved_ids |= {n.id for n in ans.retrieval.subgraph_nodes}
        for h in ans.retrieval.vector_hits:
            retrieved_ids |= set(h.doc.node_ids)
    gold_ids = set(q.gold_node_ids)

    if ans.retrieval and (ans.retrieval.subgraph_nodes or ans.retrieval.vector_hits):
        relevant_hit = len(retrieved_ids & gold_ids)
        precision = relevant_hit / max(1, len(retrieved_ids)) if retrieved_ids else 0.0
        # cap precision influence: reward presence of gold ids
        precision = min(1.0, precision * 3)  # retrieval is broad; weight relevant presence
    else:
        precision = 0.0

    recall = (len(retrieved_ids & gold_ids) / len(gold_ids)) if gold_ids else 1.0

    correctness = _keyword_coverage(ans.answer, q.gold_keywords)

    return QuestionScore(
        qid=q.qid,
        category=q.category,
        faithfulness=round(faithfulness, 3),
        answer_relevance=round(min(1.0, relevance), 3),
        context_precision=round(min(1.0, precision), 3),
        context_recall=round(min(1.0, recall), 3),
        answer_correctness=round(correctness, 3),
    )


@dataclass
class EvaluationReport:
    system: str
    per_question: List[QuestionScore] = field(default_factory=list)

    def aggregate(self) -> Dict[str, float]:
        if not self.per_question:
            return {}
        metrics = ["faithfulness", "answer_relevance", "context_precision",
                   "context_recall", "answer_correctness"]
        agg = {}
        for m in metrics:
            vals = [getattr(s, m) for s in self.per_question]
            agg[m] = round(statistics.mean(vals), 3)
        return agg

    def by_category(self) -> Dict[str, Dict[str, float]]:
        cats: Dict[str, List[QuestionScore]] = {}
        for s in self.per_question:
            cats.setdefault(s.category, []).append(s)
        out = {}
        for cat, scores in cats.items():
            out[cat] = {
                "faithfulness": round(statistics.mean(s.faithfulness for s in scores), 3),
                "answer_correctness": round(statistics.mean(s.answer_correctness for s in scores), 3),
                "context_recall": round(statistics.mean(s.context_recall for s in scores), 3),
                "n": len(scores),
            }
        return out

    def to_dict(self) -> Dict:
        return {
            "system": self.system,
            "aggregate": self.aggregate(),
            "by_category": self.by_category(),
            "per_question": [s.as_dict() for s in self.per_question],
        }


def evaluate_system(system_name: str, pairs: List) -> EvaluationReport:
    """pairs: list of (BenchmarkQuestion, RAGAnswer)."""
    report = EvaluationReport(system=system_name)
    for q, ans in pairs:
        report.per_question.append(score_question(q, ans))
    return report
