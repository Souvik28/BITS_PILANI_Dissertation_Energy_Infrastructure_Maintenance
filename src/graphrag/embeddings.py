"""Embedding backends.

* ``SentenceTransformerEmbedder`` — real dense embeddings (all-MiniLM-L6-v2 / BGE etc.)
  when ``sentence-transformers`` is installed.
* ``HashingEmbedder`` — deterministic, dependency-free fallback. A hashed bag-of-words
  projected into a fixed-dim L2-normalised vector. Not as strong as a neural model but
  good enough to make the whole pipeline runnable and the retrieval demonstrably work.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List

from .config import EmbeddingConfig

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BaseEmbedder:
    dim: int

    def embed(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover - interface
        raise NotImplementedError

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class HashingEmbedder(BaseEmbedder):
    """Deterministic hashing embedder — no third-party dependencies."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _hash(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(h, 16)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = _tokenize(text)
            for tok in tokens:
                hv = self._hash(tok)
                idx = hv % self.dim
                sign = 1.0 if (hv >> 1) % 2 == 0 else -1.0
                vec[idx] += sign
                # add a bigram-ish spread for a touch more signal
                idx2 = (hv // self.dim) % self.dim
                vec[idx2] += sign * 0.5
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


class SentenceTransformerEmbedder(BaseEmbedder):
    """Dense neural embeddings via sentence-transformers."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        embs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, row)) for row in embs]


def get_embedder(cfg: EmbeddingConfig) -> BaseEmbedder:
    """Select an embedder per config, falling back to hashing if ST is unavailable."""
    if cfg.backend in ("auto", "sentence-transformers", "st"):
        try:
            emb = SentenceTransformerEmbedder(cfg.model_name)
            print(f"[embeddings] Using sentence-transformers '{cfg.model_name}' (dim={emb.dim}).")
            return emb
        except Exception as exc:
            if cfg.backend != "auto":
                raise
            print(f"[embeddings] sentence-transformers unavailable ({exc!s}); using hashing fallback.")
    return HashingEmbedder(dim=cfg.dim)
