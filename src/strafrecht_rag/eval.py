"""Evaluation: recall@k, MRR, source correctness and no-source correctness.

The question set lives in ``tests/fixtures/eval_vragen.json``: questions with expected
ECLIs, optional filters, and questions that must hit the no-source path. Retrieval
metrics need no LLM; the generation metrics use whatever LLM backend is configured.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from strafrecht_rag.index.store import Filters
from strafrecht_rag.query.answer import Antwoord, GeenBron
from strafrecht_rag.query.generator import Generator, verify

EVAL_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eval_vragen.json"


@dataclass(frozen=True)
class EvalVraag:
    id: str
    vraag: str
    type: str
    verwacht: list[str]
    geen_bron: bool = False
    filters: Filters = field(default_factory=Filters)

    @classmethod
    def from_dict(cls, raw: dict) -> EvalVraag:
        f = raw.get("filters") or {}
        return cls(
            id=raw["id"],
            vraag=raw["vraag"],
            type=raw.get("type", "semantisch"),
            verwacht=list(raw.get("verwacht", [])),
            geen_bron=bool(raw.get("geen_bron", False)),
            filters=Filters(
                instantie_type=f.get("instantie_type"),
                instantie=f.get("instantie"),
                datum_van=date.fromisoformat(f["datum_van"]) if f.get("datum_van") else None,
                datum_tot=date.fromisoformat(f["datum_tot"]) if f.get("datum_tot") else None,
                rechtsgebied=f.get("rechtsgebied"),
            ),
        )


def load_questions(path: Path = EVAL_PATH) -> list[EvalVraag]:
    return [EvalVraag.from_dict(q) for q in json.loads(path.read_text(encoding="utf-8"))["vragen"]]


@dataclass
class VraagResultaat:
    vraag: EvalVraag
    opgehaald: list[str]  # distinct ECLIs in the top-k, in rank order
    beste_score: float
    recall: float | None  # None for no-source questions
    reciprocal_rank: float | None
    geen_bron_gegeven: bool
    geen_bron_correct: bool | None  # None when not a no-source question
    claims_raw: int = 0
    claims_ok: int = 0
    claims_afgekeurd: int = 0
    cited_eclis: list[str] = field(default_factory=list)
    cited_in_passages: bool = True
    llm_fout: str = ""  # the LLM backend failed for this question; retrieval metrics still count

    @property
    def status(self) -> str:
        if self.llm_fout:
            return "llm-fout"
        if self.vraag.geen_bron:
            return "ok" if self.geen_bron_correct else "fout"
        if self.geen_bron_gegeven:
            return "geen bron"
        return "ok" if (self.recall or 0) > 0 else "mis"


@dataclass
class EvalRapport:
    k: int
    llm: str
    resultaten: list[VraagResultaat]

    @property
    def answerable(self) -> list[VraagResultaat]:
        return [r for r in self.resultaten if not r.vraag.geen_bron]

    @property
    def no_source(self) -> list[VraagResultaat]:
        return [r for r in self.resultaten if r.vraag.geen_bron]

    def _mean(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def recall_at_k(self) -> float:
        return self._mean([r.recall for r in self.answerable if r.recall is not None])

    @property
    def hit_at_k(self) -> float:
        return self._mean([1.0 if (r.recall or 0) > 0 else 0.0 for r in self.answerable])

    @property
    def mrr(self) -> float:
        return self._mean([r.reciprocal_rank or 0.0 for r in self.answerable])

    @property
    def bronjuistheid(self) -> float:
        """Share of answers whose every cited ECLI is one of the retrieved passages."""
        answered = [r for r in self.answerable if r.cited_eclis]
        return self._mean([1.0 if r.cited_in_passages else 0.0 for r in answered])

    @property
    def claim_acceptance(self) -> float:
        raw = sum(r.claims_raw for r in self.answerable)
        return sum(r.claims_ok for r in self.answerable) / raw if raw else 0.0

    @property
    def geen_bron_correctheid(self) -> float:
        return self._mean([1.0 if r.geen_bron_correct else 0.0 for r in self.no_source])

    @property
    def onterecht_geen_bron(self) -> float:
        return self._mean([1.0 if r.geen_bron_gegeven else 0.0 for r in self.answerable])

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "llm": self.llm,
            "vragen": len(self.resultaten),
            "recall_at_k": round(self.recall_at_k, 3),
            "hit_at_k": round(self.hit_at_k, 3),
            "mrr": round(self.mrr, 3),
            "bronjuistheid": round(self.bronjuistheid, 3),
            "claim_acceptance": round(self.claim_acceptance, 3),
            "geen_bron_correctheid": round(self.geen_bron_correctheid, 3),
            "onterecht_geen_bron": round(self.onterecht_geen_bron, 3),
            "per_vraag": [
                {
                    "id": r.vraag.id,
                    "type": r.vraag.type,
                    "status": r.status,
                    "recall": r.recall,
                    "rr": r.reciprocal_rank,
                    "beste_score": round(r.beste_score, 3),
                    "opgehaald": r.opgehaald,
                    "claims_ok": r.claims_ok,
                    "claims_afgekeurd": r.claims_afgekeurd,
                    "llm_fout": r.llm_fout,
                }
                for r in self.resultaten
            ],
        }


def evaluate(
    generator: Generator,
    vragen: list[EvalVraag],
    *,
    k: int,
    with_llm: bool = True,
    on_result: Callable[[VraagResultaat], None] | None = None,
) -> EvalRapport:
    resultaten = []
    for q in vragen:
        retrieval = generator.retrieve(q.vraag, k=k, filters=q.filters)
        eclis = list(dict.fromkeys(h.ecli for h in retrieval.hits))
        expected = set(q.verwacht)
        recall = rr = None
        if not q.geen_bron and expected:
            recall = len(expected & set(eclis)) / len(expected)
            rr = next((1.0 / (i + 1) for i, e in enumerate(eclis) if e in expected), 0.0)
        result = VraagResultaat(
            vraag=q,
            opgehaald=eclis,
            beste_score=retrieval.beste_score,
            recall=recall,
            reciprocal_rank=rr,
            geen_bron_gegeven=not retrieval.boven_drempel,
            geen_bron_correct=(not retrieval.boven_drempel) if q.geen_bron else None,
        )
        if with_llm and retrieval.boven_drempel:
            try:
                answer = generator.answer_from(retrieval)
            except Exception as exc:  # one backend failure must not sink the whole run
                answer = None
                result.llm_fout = f"{type(exc).__name__}: {str(exc)[:60]}"
            if answer is None:
                pass
            elif isinstance(answer, GeenBron):
                result.geen_bron_gegeven = True
                if q.geen_bron:
                    result.geen_bron_correct = True
            elif isinstance(answer, Antwoord):
                verified = verify(answer.ruw, answer.passages)
                result.claims_ok = len(verified.beweringen) + verified.duplicaten
                result.claims_afgekeurd = len(verified.afgekeurd)
                result.claims_raw = result.claims_ok + result.claims_afgekeurd
                result.cited_eclis = answer.eclis
                passage_eclis = {p.ecli for p in answer.passages}
                result.cited_in_passages = set(answer.eclis) <= passage_eclis
        resultaten.append(result)
        if on_result:
            on_result(result)
    return EvalRapport(k=k, llm=generator.llm.name if with_llm else "—", resultaten=resultaten)
