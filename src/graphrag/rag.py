"""RAG engines: naive vector baseline and the hybrid Graph-RAG system.

Both share the same answer envelope (``RAGAnswer``) so the evaluation harness can compare
them apples-to-apples. GraphRAG additionally runs the grounding checker and can re-prompt
if the first answer is poorly grounded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config
from .grounding import GroundingChecker, GroundingReport
from .graph_store import BaseGraphStore
from .llm import BaseLLM
from .retrieval import HybridRetriever, RetrievalResult, VectorRetriever
from .vector_store import VectorStore

SYSTEM_PROMPT = (
    "You are a maintenance-engineering decision-support assistant for energy "
    "infrastructure (offshore platforms, refineries, LNG). Answer ONLY from the provided "
    "context. Cite equipment tags and SOP ids. If the context does not support an answer, "
    "say so explicitly. Pay attention to upstream/downstream flow relationships."
)

PROMPT_TEMPLATE = """{system}

CONTEXT:
{context}

QUESTION: {question}

Provide a concise, grounded answer. Reference specific equipment ids, failure codes, and
SOP ids from the context. Do not invent facts.
ANSWER:"""


@dataclass
class RAGAnswer:
    question: str
    answer: str
    mode: str
    contexts: List[str] = field(default_factory=list)
    graph_facts: List[str] = field(default_factory=list)
    grounding: Optional[GroundingReport] = None
    retrieval: Optional[RetrievalResult] = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "mode": self.mode,
            "contexts": self.contexts,
            "graph_facts": self.graph_facts,
            "grounding_score": self.grounding.score if self.grounding else None,
            "unsupported_claims": self.grounding.unsupported_claims if self.grounding else [],
        }


class NaiveVectorRAG:
    """Control baseline: chunk-based vector retrieval + generation. No graph."""

    def __init__(self, vector_store: VectorStore, llm: BaseLLM, cfg: Config):
        self.retriever = VectorRetriever(vector_store, cfg.retrieval)
        self.llm = llm
        self.cfg = cfg
        self.grounding = GroundingChecker()

    def answer(self, question: str) -> RAGAnswer:
        result = self.retriever.retrieve(question)
        prompt = PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT, context=result.context_text(), question=question
        )
        text = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        report = self.grounding.check(text, result)
        return RAGAnswer(
            question=question,
            answer=text,
            mode="naive_vector",
            contexts=result.passages,
            graph_facts=[],
            grounding=report,
            retrieval=result,
        )


class GraphRAG:
    """Hybrid Graph-RAG with grounding and optional grounded re-prompt."""

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: BaseGraphStore,
        llm: BaseLLM,
        cfg: Config,
    ):
        self.retriever = HybridRetriever(vector_store, graph_store, cfg.retrieval)
        self.llm = llm
        self.cfg = cfg
        self.grounding = GroundingChecker()

    def answer(self, question: str, reground: bool = True) -> RAGAnswer:
        result = self.retriever.retrieve(question)
        prompt = PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT, context=result.context_text(), question=question
        )
        text = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        report = self.grounding.check(text, result)

        # If poorly grounded, re-prompt with an explicit instruction to stick to facts.
        if reground and not report.is_grounded and result.graph_facts:
            stricter = prompt + (
                "\n\nIMPORTANT: Your previous answer contained unsupported claims. "
                "Re-answer using ONLY the STRUCTURED GRAPH FACTS above. If unknown, say so."
            )
            text2 = self.llm.generate(stricter, system=SYSTEM_PROMPT)
            report2 = self.grounding.check(text2, result)
            if report2.score >= report.score:
                text, report = text2, report2

        return RAGAnswer(
            question=question,
            answer=text,
            mode="graph_rag",
            contexts=result.passages,
            graph_facts=result.graph_facts,
            grounding=report,
            retrieval=result,
        )
