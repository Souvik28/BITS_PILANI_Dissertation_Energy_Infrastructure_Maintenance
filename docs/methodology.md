# Methodology & 16-Week Plan Crosswalk

This document maps the dissertation's 16-week plan to the concrete artifacts in this repo,
and explains the experimental design used to compare Naive Vector RAG against Graph-RAG.

## System overview

1. **Synthetic data generation** (`data_generation.py`) builds a coherent asset graph:
   platforms → components wired by `CONNECTED_TO` flow edges → telemetry, failure modes,
   SOPs, and historical work orders. A seed makes it fully reproducible.
2. **Ingestion** (`ingest.py`) converts the dataset into (a) a property graph and (b) a
   text corpus where every document is linked to the graph node ids it describes. This
   linkage is what enables hybrid retrieval.
3. **Retrieval** (`retrieval.py`):
   - *Vector*: semantic search over the corpus.
   - *Graph*: entity-link the query → seed nodes, then extract a multi-hop subgraph with
     explicit downstream/upstream handling.
   - *Hybrid*: vector hits seed additional graph traversal; the subgraph is rendered to
     structured "graph facts" fused with the retrieved passages.
4. **Grounding** (`grounding.py`): every answer sentence is checked for support against the
   retrieved evidence, producing a groundedness score. Poorly-grounded Graph-RAG answers
   trigger a stricter re-prompt (fact-only).
5. **Generation** (`llm.py`): Ollama when available, otherwise a deterministic extractive
   synthesiser so the pipeline and evaluation always run.
6. **Routing** (`router.py`): a transparent LangGraph-style policy selects vector / graph /
   hybrid based on query signals (topology words, document references, lookups).
7. **Evaluation** (`evaluation.py`): RAGAS-style lexical metrics against a 50-question
   gold benchmark auto-derived from the graph (`benchmark.py`).

## Experimental design

- **Control (baseline):** `NaiveVectorRAG` — chunk retrieval + generation, no graph.
- **Treatment:** `GraphRAG` — hybrid retrieval + grounding.
- **Benchmark:** 50 questions in four classes: `lookup`, `topology`, `multi_hop`,
  `procedural`. Each has a deterministic gold answer and gold evidence node ids.
- **Metrics:** faithfulness/groundedness, answer relevance, context precision, context
  recall, answer correctness.
- **Hypothesis:** Graph-RAG improves correctness and recall most on `topology` and
  `multi_hop` questions, where naive vector retrieval cannot follow the asset flow graph.

Run `python -m scripts.run_evaluation` to reproduce the comparison; results are written to
`data/results/evaluation_latest.json`.

## Plan crosswalk

| Weeks | Plan item                                   | Artifact(s)                                            |
|-------|---------------------------------------------|--------------------------------------------------------|
| 1     | Literature survey, scope, metric defs       | this file, `evaluation.py` metric definitions          |
| 2     | ISO 14224 schema, property-graph ontology   | `schema.py`, `docs/ontology.md`                        |
| 3     | Data harvesting + synthetic generation      | `data_generation.py`                                   |
| 4     | KB validation + 50-question benchmark       | `ingest.py`, `benchmark.py`                            |
| 5     | Graph DB infrastructure (Neo4j/Docker)      | `graph_store.py`, `docker/docker-compose.yml`          |
| 6     | Entity/relation extraction (KG construction)| `ingest.build_graph`, LLM-assisted edges via `llm.py`  |
| 7     | Hybrid retrieval (vector + graph)           | `vector_store.py`, `retrieval.py`                      |
| 8     | Graph-RAG synthesis + grounding module      | `rag.GraphRAG`, `grounding.py`                         |
| 9     | Baseline naive vector RAG                    | `rag.NaiveVectorRAG`                                   |
| 10    | Interface + routing agent                   | `app/streamlit_app.py`, `router.py`                    |
| 11    | RAG evaluation framework                     | `evaluation.py`, `scripts/run_evaluation.py`          |
| 12    | Quantitative + qualitative analysis          | eval by-category breakdown, `data/results/`           |
| 13    | Tuning + optimization                        | retrieval config knobs in `config.py`, re-prompt loop |
| 14-16 | Thesis writing, demonstration, defense       | `docs/`, Streamlit demo                                |

## Notes on real datasets (Volve / NASA PCoE / MIMII)

The synthetic generator mirrors the structure of these open datasets (telemetry channels:
vibration, temperature, pressure, flow). To ingest real data, add a loader that maps
source rows onto the `schema.py` dataclasses and feed them through `ingest.build_graph` /
`build_corpus`. The rest of the pipeline is dataset-agnostic.
