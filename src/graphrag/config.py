"""Central configuration for the Graph-RAG pipeline.

Every tunable lives here. Backends are chosen via environment variables so the same code
runs on a laptop (zero deps) or against Neo4j + a local LLM without edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GRAPHRAG_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = DATA_DIR / "figures"

DATASET_FILE = PROCESSED_DIR / "energy_infrastructure_dataset.json"
CORPUS_FILE = PROCESSED_DIR / "text_corpus.json"
GRAPH_FILE = PROCESSED_DIR / "knowledge_graph.json"
BENCHMARK_FILE = PROCESSED_DIR / "benchmark.json"
VECTOR_INDEX_FILE = INDEX_DIR / "vector_index.json"


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, INDEX_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


@dataclass
class GraphConfig:
    """Graph backend selection. ``backend`` is 'networkx' or 'neo4j'."""

    backend: str = os.environ.get("GRAPHRAG_GRAPH", "networkx").lower()
    neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "graphrag_pw")
    neo4j_database: str = os.environ.get("NEO4J_DATABASE", "neo4j")
    # If neo4j is requested but unavailable, silently fall back to networkx.
    allow_fallback: bool = _env_bool("GRAPHRAG_GRAPH_FALLBACK", True)


@dataclass
class EmbeddingConfig:
    """Embedding backend. 'auto' uses sentence-transformers if installed."""

    backend: str = os.environ.get("GRAPHRAG_EMBED", "auto").lower()
    model_name: str = os.environ.get("GRAPHRAG_EMBED_MODEL", "all-MiniLM-L6-v2")
    dim: int = _env_int("GRAPHRAG_EMBED_DIM", 256)  # used by hashing fallback only


@dataclass
class LLMConfig:
    """LLM backend. 'auto' uses Ollama if reachable, else extractive synthesis."""

    backend: str = os.environ.get("GRAPHRAG_LLM", "auto").lower()
    model: str = os.environ.get("GRAPHRAG_LLM_MODEL", "llama3")
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    temperature: float = float(os.environ.get("GRAPHRAG_LLM_TEMP", "0.1"))
    max_tokens: int = _env_int("GRAPHRAG_LLM_MAX_TOKENS", 768)
    request_timeout: int = _env_int("GRAPHRAG_LLM_TIMEOUT", 60)


@dataclass
class RetrievalConfig:
    top_k_vector: int = _env_int("GRAPHRAG_TOPK_VECTOR", 6)
    graph_hops: int = _env_int("GRAPHRAG_GRAPH_HOPS", 2)
    max_graph_nodes: int = _env_int("GRAPHRAG_MAX_GRAPH_NODES", 40)
    hybrid_vector_weight: float = float(os.environ.get("GRAPHRAG_HYBRID_W", "0.5"))


@dataclass
class DataGenConfig:
    seed: int = _env_int("GRAPHRAG_SEED", 42)
    num_assets: int = _env_int("GRAPHRAG_NUM_ASSETS", 4)
    comps_per_asset: int = _env_int("GRAPHRAG_COMPS_PER_ASSET", 14)
    work_orders: int = _env_int("GRAPHRAG_WORK_ORDERS", 240)
    telemetry_per_comp: int = _env_int("GRAPHRAG_TELEMETRY_PER_COMP", 8)


@dataclass
class Config:
    graph: GraphConfig = field(default_factory=GraphConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    datagen: DataGenConfig = field(default_factory=DataGenConfig)


CONFIG = Config()
