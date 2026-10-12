"""Answer generation with source obligation, enforced in code (PRD N2).

The pipeline is three explicit steps so each one is testable and visible:

1. ``retrieve`` — below the score threshold the LLM is never called: "no source" is a
   code path, not prompt behaviour.
2. ``generate`` — the model gets numbered passages and must end every claim with
   ``[n]`` markers. It may answer ``GEEN_BRON`` when the passages do not answer the question.
3. ``verify`` — every claim is checked in code: markers must point at passages that were
   actually supplied, and any ECLI mentioned in the text must be one of those passages'
   ECLIs. A claim that fails is rejected and reported, never silently kept.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from strafrecht_rag.index.store import Filters, Hit
from strafrecht_rag.query.answer import Afgekeurd, Antwoord, Bewering, GeenBron, Verwijzing
from strafrecht_rag.query.llm import LLM
from strafrecht_rag.query.retriever import Retrieval, Retriever

MARKER_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
ECLI_RE = re.compile(r"ECLI:[A-Z]{2}:[A-Z0-9]+:\d{4}:[A-Z0-9.]+")
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
NO_SOURCE_TOKEN = "GEEN_BRON"

SYSTEM_PROMPT = """Je bent een juridisch zoekassistent voor Nederlandse strafrechtelijke uitspraken.

Je krijgt een vraag en een aantal genummerde passages uit uitspraken. Regels:
1. Antwoord UITSLUITEND op basis van de passages. Gebruik geen eigen kennis.
2. Schrijf korte, feitelijke beweringen in het Nederlands, elke bewering op een eigen regel.
3. Eindig ELKE regel met de nummers van de passages waarop die regel steunt, zoals [1] of [2][4].
   Een regel zonder verwijzing wordt weggegooid.
4. Noem geen ECLI-nummers die niet in de passages staan.
5. Als de passages de vraag niet beantwoorden, antwoord dan met precies één woord: GEEN_BRON
6. Geen inleiding, geen samenvatting achteraf, geen juridisch advies.
7. Maximaal 6 regels; herhaal een bewering niet.

Vorm van elke regel: <één feitelijke bewering uit de passages> [<passagenummer>]
Bijvoorbeeld: <bewering> [2] of <bewering> [1][3]. Herhaal dit voorbeeld niet letterlijk."""

RETRY_REMINDER = (
    "\n\nLet op: je vorige antwoord bevatte geen geldige verwijzingen. Elke regel MOET eindigen "
    "met het nummer van de passage tussen blokhaken, bijvoorbeeld [1]. Probeer het opnieuw."
)


def format_passages(hits: list[Hit]) -> str:
    blocks = []
    for n, hit in enumerate(hits, 1):
        header = (
            f"[{n}] {hit.ecli} · {hit.instantie} · {hit.datum} · "
            f"§{hit.sectie_nummer} {hit.sectie_titel}"
        )
        blocks.append(f"{header}\n{hit.tekst}")
    return "\n\n".join(blocks)


def build_prompt(vraag: str, hits: list[Hit]) -> str:
    return f"Vraag: {vraag}\n\nPassages:\n\n{format_passages(hits)}\n\n---\nAntwoord:"


@dataclass(frozen=True)
class Verificatie:
    beweringen: list[Bewering]
    afgekeurd: list[Afgekeurd]
    geen_bron: bool = False
    duplicaten: int = 0  # repeated claims collapsed into one (small models loop)


def _key(tekst: str) -> str:
    return re.sub(r"[^\w]+", " ", tekst.lower()).strip()


def _split_markers(text: str) -> tuple[str, list[int]]:
    numbers: list[int] = []
    for group in MARKER_RE.findall(text):
        numbers.extend(int(x) for x in re.split(r"\s*[,;]\s*", group))
    clean = MARKER_RE.sub("", text)
    clean = re.sub(r"\s+", " ", clean).strip(" .:;,") + "."
    return clean, list(dict.fromkeys(numbers))


def _claims(raw: str) -> Iterator[str]:
    for line in raw.splitlines():
        line = BULLET_RE.sub("", line).strip()
        if any(ch.isalpha() for ch in line):  # blank lines and stray punctuation are not claims
            yield line


def verify(raw: str, hits: list[Hit]) -> Verificatie:
    """Check every claim against the supplied passages; reject what cannot be traced."""
    if raw.strip().upper().rstrip(".") == NO_SOURCE_TOKEN:
        return Verificatie([], [], geen_bron=True)
    allowed_eclis = {h.ecli for h in hits}
    beweringen: list[Bewering] = []
    afgekeurd: list[Afgekeurd] = []
    duplicaten = 0
    for claim in _claims(raw):
        if claim.upper().rstrip(".") == NO_SOURCE_TOKEN:
            continue
        tekst, numbers = _split_markers(claim)
        if not numbers:
            afgekeurd.append(Afgekeurd(tekst, "geen verwijzing naar een passage"))
            continue
        invalid = [n for n in numbers if not 1 <= n <= len(hits)]
        if invalid:
            afgekeurd.append(
                Afgekeurd(tekst, f"verwijzing naar niet-aangeleverde passage {invalid}")
            )
            continue
        unknown = sorted(set(ECLI_RE.findall(tekst)) - allowed_eclis)
        if unknown:
            afgekeurd.append(Afgekeurd(tekst, f"ECLI niet in de passages: {', '.join(unknown)}"))
            continue
        if len(tekst) < 4:
            afgekeurd.append(Afgekeurd(tekst, "lege bewering"))
            continue
        refs = [Verwijzing.from_hit(n, hits[n - 1]) for n in numbers]
        earlier = next((b for b in beweringen if _key(b.tekst) == _key(tekst)), None)
        if earlier is not None:  # same claim again: keep it once, merge its references
            duplicaten += 1
            merged = list(earlier.verwijzingen) + [r for r in refs if r not in earlier.verwijzingen]
            beweringen[beweringen.index(earlier)] = Bewering(earlier.tekst, merged)
            continue
        beweringen.append(Bewering(tekst, refs))
    return Verificatie(beweringen, afgekeurd, duplicaten=duplicaten)


class Generator:
    def __init__(self, retriever: Retriever, llm: LLM) -> None:
        self.retriever = retriever
        self.llm = llm

    def retrieve(
        self, vraag: str, *, k: int | None = None, filters: Filters | None = None, on_stage=None
    ):
        return self.retriever.retrieve(vraag, k=k, filters=filters, on_stage=on_stage)

    def generate(self, vraag: str, retrieval: Retrieval) -> str:
        """One call; if nothing survives verification, one retry with a format reminder.

        Both attempts are kept in the raw output (separated by a marker) so the eval and
        the ``--json`` output show what really happened.
        """
        raw = self.llm.complete(SYSTEM_PROMPT, build_prompt(vraag, retrieval.hits))
        verified = verify(raw, retrieval.hits)
        if verified.beweringen or verified.geen_bron:
            return raw
        prompt = build_prompt(vraag, retrieval.hits) + RETRY_REMINDER
        return self.llm.complete(SYSTEM_PROMPT, prompt)

    def answer(
        self, vraag: str, *, k: int | None = None, filters: Filters | None = None
    ) -> Antwoord | GeenBron:
        retrieval = self.retrieve(vraag, k=k, filters=filters)
        return self.answer_from(retrieval)

    def answer_from(self, retrieval: Retrieval, raw: str | None = None) -> Antwoord | GeenBron:
        if not retrieval.boven_drempel:
            return GeenBron(
                retrieval.vraag,
                reden=(
                    f"beste score {retrieval.beste_score:.2f} ligt onder de drempel "
                    f"{retrieval.drempel:.2f}; het taalmodel is niet aangeroepen"
                ),
                beste_score=retrieval.beste_score,
                drempel=retrieval.drempel,
                kandidaten=retrieval.dichtstbij,
            )
        raw = self.generate(retrieval.vraag, retrieval) if raw is None else raw
        verified = verify(raw, retrieval.hits)
        if verified.geen_bron:
            return GeenBron(
                retrieval.vraag,
                reden="het taalmodel vond in de aangeleverde passages geen antwoord (GEEN_BRON)",
                beste_score=retrieval.beste_score,
                drempel=retrieval.drempel,
                kandidaten=retrieval.hits,
            )
        return Antwoord(
            vraag=retrieval.vraag,
            beweringen=verified.beweringen,
            afgekeurd=verified.afgekeurd,
            passages=retrieval.hits,
            llm=self.llm.name,
            ruw=raw,
            duplicaten=verified.duplicaten,
        )
