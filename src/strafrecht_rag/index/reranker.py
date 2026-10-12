"""Cross-encoder reranking: a calibrated relevance score for the top candidates.

Bi-encoder cosine similarity ranks well but is not calibrated: with a small multilingual
embedder, an unrelated question still scores 0.6 against some Dutch legal paragraph. A
cross-encoder reads question and passage together and answers "is this relevant?"; its
scores separate real hits from noise, which is what the no-source threshold needs.

Rerankers return probabilities in [0, 1] (sigmoid of the model logit).
"""

from __future__ import annotations

import math
import os
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from strafrecht_rag.index.stopwords import STOPWORDS_NL

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class FakeReranker:
    """Deterministic: share of the query's content words that occur in the passage."""

    name = "fake"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        terms = {t for t in TOKEN_RE.findall(query.lower()) if t not in STOPWORDS_NL and len(t) > 2}
        out = []
        for text in texts:
            words = set(TOKEN_RE.findall(text.lower()))
            out.append(len(terms & words) / len(terms) if terms else 0.0)
        return out


class FastEmbedReranker:
    def __init__(self, model: str) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed"))
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = TextCrossEncoder(model_name=model)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        return [_sigmoid(float(logit)) for logit in self._model.rerank(query, list(texts))]


def make_reranker(model: str | None) -> Reranker | None:
    if not model or model.lower() in {"", "none", "off"}:
        return None
    if model == "fake":
        return FakeReranker()
    return FastEmbedReranker(model)
