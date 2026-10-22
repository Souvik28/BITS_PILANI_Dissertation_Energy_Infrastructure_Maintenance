"""Tests for benchmark generation and evaluation metrics."""
from graphrag.benchmark import build_benchmark
from graphrag.config import CONFIG
from graphrag.evaluation import evaluate_system, score_question
from graphrag.llm import ExtractiveLLM
from graphrag.rag import GraphRAG, NaiveVectorRAG
from graphrag.vector_store import VectorStore
from graphrag.embeddings import HashingEmbedder


def test_benchmark_generates_questions(graph, small_cfg):
    bench = build_benchmark(graph, small_cfg, target=50)
    assert 10 <= len(bench) <= 50
    cats = {q.category for q in bench}
    # should span multiple categories
    assert len(cats) >= 3
    for q in bench:
        assert q.question and q.gold_answer
        assert q.qid.startswith("Q")


def test_evaluation_produces_metrics(graph, corpus, small_cfg):
    emb = HashingEmbedder(dim=256)
    vs = VectorStore(emb)
    vs.add_documents(corpus)
    llm = ExtractiveLLM()
    naive = NaiveVectorRAG(vs, llm, CONFIG)
    graphrag = GraphRAG(vs, graph, llm, CONFIG)

    bench = build_benchmark(graph, small_cfg, target=12)
    naive_pairs = [(q, naive.answer(q.question)) for q in bench]
    graph_pairs = [(q, graphrag.answer(q.question)) for q in bench]

    nr = evaluate_system("naive", naive_pairs)
    gr = evaluate_system("graph", graph_pairs)

    for report in (nr, gr):
        agg = report.aggregate()
        for metric in ("faithfulness", "context_recall", "answer_correctness"):
            assert 0.0 <= agg[metric] <= 1.0


def test_score_question_ranges(graph, corpus, small_cfg):
    emb = HashingEmbedder(dim=128)
    vs = VectorStore(emb)
    vs.add_documents(corpus)
    graphrag = GraphRAG(vs, graph, ExtractiveLLM(), CONFIG)
    bench = build_benchmark(graph, small_cfg, target=5)
    q = bench[0]
    ans = graphrag.answer(q.question)
    s = score_question(q, ans)
    for v in (s.faithfulness, s.answer_relevance, s.context_precision,
              s.context_recall, s.answer_correctness):
        assert 0.0 <= v <= 1.0
