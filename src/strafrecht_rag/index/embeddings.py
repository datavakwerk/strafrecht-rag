"""Embeddings behind one small interface.

* ``FakeEmbedder`` — deterministic hashed bag-of-words vectors: tests and CI run on it,
  no model download, no network (PRD N1). Similar texts still land close together, so
  retrieval tests are meaningful, not vacuous.
* ``FastEmbedEmbedder`` — local ONNX model via fastembed (default:
  ``paraphrase-multilingual-MiniLM-L12-v2``); the choice is in the README decisions log.
* ``OllamaEmbedder`` — any embedding model served by a local Ollama.
"""

from __future__ import annotations

import hashlib
import os
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from strafrecht_rag.index.stopwords import STOPWORDS_NL

Kind = Literal["document", "query"]
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str], *, kind: Kind = "document") -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class FakeEmbedder:
    """Hashed bag-of-words. Same text → same vector; shared words → higher cosine."""

    name = "fake"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str], *, kind: Kind = "document") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN_RE.findall(text.lower()):
                if token in STOPWORDS_NL:
                    continue
                digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
                out[row, int.from_bytes(digest, "little") % self.dim] += 1.0
        return _normalize(out)


class FastEmbedEmbedder:
    def __init__(self, model: str, *, batch_size: int = 64, threads: int | None = None) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed"))
        from fastembed import TextEmbedding

        self.name = model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # fastembed's pooling-change notice, once per process
            self._model = TextEmbedding(model_name=model, threads=threads)
        self._batch = batch_size
        self._e5 = "e5" in model.lower()
        self.dim = int(self.embed(["dim probe"]).shape[1])

    def embed(self, texts: Sequence[str], *, kind: Kind = "document") -> np.ndarray:
        if self._e5:  # e5 models are trained with these prefixes
            prefix = "query: " if kind == "query" else "passage: "
            texts = [prefix + t for t in texts]
        vectors = list(self._model.embed(list(texts), batch_size=self._batch))
        return _normalize(np.stack(vectors)) if vectors else np.zeros((0, self.dim), np.float32)


class OllamaEmbedder:
    def __init__(self, model: str, url: str = "http://localhost:11434") -> None:
        import httpx

        self.name = f"ollama/{model}"
        self._model = model
        self._client = httpx.Client(base_url=url, timeout=120.0)
        self.dim = int(self.embed(["dim probe"]).shape[1])

    def embed(self, texts: Sequence[str], *, kind: Kind = "document") -> np.ndarray:
        rows = []
        for start in range(0, len(texts), 32):
            response = self._client.post(
                "/api/embed", json={"model": self._model, "input": list(texts[start : start + 32])}
            )
            response.raise_for_status()
            rows.extend(response.json()["embeddings"])
        return _normalize(np.array(rows, dtype=np.float32))


def make_embedder(backend: str, model: str, *, ollama_url: str = "http://localhost:11434"):
    match backend:
        case "fake":
            return FakeEmbedder()
        case "fastembed":
            return FastEmbedEmbedder(model)
        case "ollama":
            return OllamaEmbedder(model, ollama_url)
    raise ValueError(f"unknown embedder backend {backend!r}")
