"""Vector index with FAISS when available, numpy/pure-python fallback otherwise.

Stores embedding vectors alongside the ``TextDoc`` payloads so a search returns both the
score and the document (including its linked graph ``node_ids``). Persists to a JSON file
so indexes survive between CLI invocations without extra services.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .embeddings import BaseEmbedder
from .ingest import TextDoc


@dataclass
class SearchHit:
    doc: TextDoc
    score: float


class VectorStore:
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self.docs: List[TextDoc] = []
        self.vectors: List[List[float]] = []
        self._faiss_index = None
        self._np = None

    # ------------------------------------------------------------------ build
    def add_documents(self, docs: List[TextDoc]) -> None:
        texts = [d.text for d in docs]
        vecs = self.embedder.embed(texts)
        self.docs.extend(docs)
        self.vectors.extend(vecs)
        self._faiss_index = None  # invalidate; rebuilt lazily

    def _try_build_faiss(self) -> bool:
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return False
        if not self.vectors:
            return False
        self._np = np
        mat = np.array(self.vectors, dtype="float32")
        # vectors are L2-normalised -> inner product == cosine similarity
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        self._faiss_index = index
        return True

    # ------------------------------------------------------------------ search
    def search(self, query: str, top_k: int = 6) -> List[SearchHit]:
        if not self.vectors:
            return []
        qvec = self.embedder.embed_one(query)

        if self._faiss_index is None:
            self._try_build_faiss()

        if self._faiss_index is not None and self._np is not None:
            q = self._np.array([qvec], dtype="float32")
            scores, idxs = self._faiss_index.search(q, min(top_k, len(self.docs)))
            return [
                SearchHit(doc=self.docs[i], score=float(s))
                for s, i in zip(scores[0], idxs[0])
                if i >= 0
            ]

        # pure-python cosine (vectors already normalised)
        scored: List[Tuple[float, int]] = []
        for i, vec in enumerate(self.vectors):
            dot = sum(a * b for a, b in zip(qvec, vec))
            scored.append((dot, i))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [SearchHit(doc=self.docs[i], score=float(s)) for s, i in scored[:top_k]]

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.embedder.dim,
            "docs": [d.to_dict() for d in self.docs],
            "vectors": self.vectors,
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: Path, embedder: BaseEmbedder) -> "VectorStore":
        with open(path) as f:
            payload = json.load(f)
        store = cls(embedder)
        store.docs = [TextDoc.from_dict(d) for d in payload["docs"]]
        store.vectors = payload["vectors"]
        return store
