"""Retrieval: embed the question, take top-k with filters, apply the score threshold.

Hybrid by default: cosine top-2k fused with BM25 top-2k by reciprocal rank fusion.
The threshold is on the *cosine* of the best hit, so "no source" is a property of the
retrieval, not of whatever the LLM feels like saying (the no-source path of the
generator rests on this).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from strafrecht_rag.index.embeddings import Embedder
from strafrecht_rag.index.reranker import Reranker
from strafrecht_rag.index.store import Filters, Hit, VectorStore

RRF_K = 60


@dataclass(frozen=True)
class Retrieval:
    vraag: str
    hits: list[Hit]  # ordered, already cut to k; empty when below the threshold
    beste_score: float
    drempel: float
    filters: Filters = field(default_factory=Filters)
    kandidaten: int = 0  # how many distinct chunks the two searches produced together
    score_soort: str = "cosine"  # what beste_score / drempel are measured in
    herordend: int = 0  # candidates that went through the reranker
    dichtstbij: list[Hit] = field(default_factory=list)  # best candidates when below threshold

    @property
    def boven_drempel(self) -> bool:
        return bool(self.hits)


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        k: int = 6,
        threshold: float = 0.35,
        hybrid: bool = True,
        per_ecli: int = 2,
        reranker: Reranker | None = None,
        rerank_top: int = 12,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.k = k
        self.threshold = threshold
        self.hybrid = hybrid
        self.per_ecli = per_ecli
        self.reranker = reranker
        self.rerank_top = rerank_top

    def retrieve(
        self,
        vraag: str,
        *,
        k: int | None = None,
        filters: Filters | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> Retrieval:
        """``on_stage`` is called with "embed", "search" and "rerank" as each step starts."""
        k = k or self.k
        filters = filters or Filters()
        stage = on_stage or (lambda _name: None)
        stage("embed")
        vector = self.embedder.embed([vraag], kind="query")[0]
        stage("search")
        dense = self.store.search_vector(vector, k * 3, filters)
        sparse = self.store.search_text(vraag, vector, k * 3, filters) if self.hybrid else []
        fused = _rrf(dense, sparse)
        soort, herordend = "cosine", 0
        if self.reranker is not None and fused:
            stage("rerank")
            fused = self._rerank(vraag, fused[: self.rerank_top])
            soort, herordend = "rerank", len(fused)
        beste = max((_relevance(h) for h in fused), default=0.0)
        if beste < self.threshold:
            return Retrieval(
                vraag, [], beste, self.threshold, filters, len(fused), soort, herordend, fused[:3]
            )
        # what scores under half the threshold is noise for the model, not a source
        usable = [h for h in fused if _relevance(h) >= self.threshold / 2]
        hits = _diversify(usable, k, self.per_ecli)
        return Retrieval(vraag, hits, beste, self.threshold, filters, len(fused), soort, herordend)

    def _rerank(self, vraag: str, hits: list[Hit]) -> list[Hit]:
        scores = self.reranker.score(vraag, [f"{h.sectie_titel}\n{h.tekst}" for h in hits])
        scored = [replace(h, relevantie=s) for h, s in zip(hits, scores, strict=True)]
        return sorted(scored, key=lambda h: (-(h.relevantie or 0.0), -h.score))


def _relevance(hit: Hit) -> float:
    return hit.relevantie if hit.relevantie is not None else hit.score


def _rrf(dense: list[Hit], sparse: list[Hit]) -> list[Hit]:
    scores: dict[str, float] = {}
    hits: dict[str, Hit] = {}
    for rank, hit in enumerate(dense, 1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        hits[hit.chunk_id] = hit
    for rank, hit in enumerate(sparse, 1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        if hit.chunk_id in hits:
            hits[hit.chunk_id] = Hit(**{**hits[hit.chunk_id].__dict__, "bm25_rank": rank})
        else:
            hits[hit.chunk_id] = hit
    ordered = sorted(scores, key=lambda cid: (-scores[cid], -hits[cid].score))
    return [hits[cid] for cid in ordered]


def _diversify(hits: list[Hit], k: int, per_ecli: int) -> list[Hit]:
    """At most ``per_ecli`` chunks per ruling in the top-k, so 'which rulings…' gets breadth."""
    out: list[Hit] = []
    seen: dict[str, int] = {}
    for hit in hits:
        if seen.get(hit.ecli, 0) >= per_ecli:
            continue
        seen[hit.ecli] = seen.get(hit.ecli, 0) + 1
        out.append(hit)
        if len(out) == k:
            break
    return out
