"""LLM abstraction.

* ``OllamaLLM``        — talks to a local Ollama server (llama3 / mistral / etc.) over HTTP
  using only the standard library (``urllib``), so no python SDK is required.
* ``ExtractiveLLM``    — dependency-free fallback that synthesises an answer directly from
  the retrieved context (ranked sentence extraction). Keeps the whole pipeline runnable
  and the evaluation meaningful even with no model available.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import List, Optional

from .config import LLMConfig

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]+")


class BaseLLM:
    name: str = "base"

    def generate(self, prompt: str, system: Optional[str] = None) -> str:  # pragma: no cover
        raise NotImplementedError


class OllamaLLM(BaseLLM):
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.name = f"ollama:{cfg.model}"
        self._check()

    def _check(self) -> None:
        req = urllib.request.Request(f"{self.cfg.ollama_host}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            json.loads(resp.read().decode("utf-8"))

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": self.cfg.temperature, "num_predict": self.cfg.max_tokens},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.cfg.ollama_host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.cfg.request_timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "").strip()


class ExtractiveLLM(BaseLLM):
    """Ranks context sentences by query overlap and composes a grounded answer."""

    name = "extractive-fallback"

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        # The question is the first line of the QUESTION section (the rest is instructions).
        question = self._extract_section(prompt, "QUESTION:").splitlines()[0] \
            if self._extract_section(prompt, "QUESTION:") else ""
        context = self._extract_section(prompt, "CONTEXT:") or prompt
        q_tokens = {t.lower() for t in _TOKEN_RE.findall(question)}

        # Only keep genuine evidence lines (the context renders them as "- ..." bullets).
        # This avoids echoing the question or the instruction text back as an answer.
        sentences: List[str] = []
        for line in context.splitlines():
            stripped = line.strip()
            if not stripped.startswith("-"):
                continue
            # drop section headers like "STRUCTURED GRAPH FACTS:" that have no bullet
            body = stripped.lstrip("-").strip()
            for s in _SENT_SPLIT.split(body):
                s = s.strip()
                if len(s) > 10:
                    sentences.append(s)

        scored = []
        for s in sentences:
            stoks = {t.lower() for t in _TOKEN_RE.findall(s)}
            overlap = len(q_tokens & stoks)
            scored.append((overlap, s))
        scored.sort(key=lambda t: t[0], reverse=True)

        picked: List[str] = []
        seen = set()
        for score, s in scored:
            if score == 0 and picked:
                break
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append(s)
            if len(picked) >= 6:
                break

        if not picked:
            return "No grounded answer could be produced from the retrieved context."
        return (
            "Based on the retrieved knowledge graph and documents: "
            + " ".join(s if s.endswith((".", "!", "?")) else s + "." for s in picked)
        )

    # Top-level prompt headers that delimit sections (ordered as they appear).
    _SECTION_HEADERS = ("CONTEXT:", "QUESTION:", "ANSWER:")

    @classmethod
    def _extract_section(cls, text: str, header: str) -> str:
        idx = text.find(header)
        if idx < 0:
            return ""
        start = idx + len(header)
        rest = text[start:]
        # Cut at the next *top-level* header only (not inner caps like
        # "STRUCTURED GRAPH FACTS:"), so context keeps all its evidence bullets.
        cut = len(rest)
        for h in cls._SECTION_HEADERS:
            if h == header:
                continue
            pos = rest.find(h)
            if 0 <= pos < cut:
                cut = pos
        return rest[:cut].strip()


def get_llm(cfg: LLMConfig) -> BaseLLM:
    if cfg.backend in ("auto", "ollama"):
        try:
            llm = OllamaLLM(cfg)
            print(f"[llm] Using Ollama model '{cfg.model}'.")
            return llm
        except Exception as exc:
            if cfg.backend != "auto":
                raise
            print(f"[llm] Ollama unavailable ({exc!s}); using extractive fallback.")
    return ExtractiveLLM()
