"""Chunking: pure, deterministic, and traceable.

A chunk never crosses a section boundary, carries the paragraph range it was cut
from, and overlaps its predecessor by whole paragraphs (or whole sentence pieces
when a single paragraph is longer than a chunk). Sizes are in characters: model
independent, reproducible, and good enough for the tuning knob this is.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from strafrecht_rag.ingest.model import Sectie, Uitspraak

SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+(?=\S)")


@dataclass(frozen=True)
class ChunkConfig:
    size: int = 900  # target maximum characters per chunk
    overlap: int = 150  # characters carried over from the previous chunk
    min_chars: int = 20  # drop crumbs shorter than this

    def __post_init__(self) -> None:
        if self.size <= 0 or self.overlap < 0 or self.overlap >= self.size:
            raise ValueError("chunk size must be > 0 and overlap in [0, size)")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<ECLI>#<positie>": stable, so re-indexing is an upsert
    ecli: str
    positie: int  # 0-based index within the ruling
    sectie_nummer: str
    sectie_titel: str
    para_van: int
    para_tot: int
    tekst: str

    @property
    def bron(self) -> str:
        paras = (
            f"par. {self.para_van}"
            if self.para_van == self.para_tot
            else f"par. {self.para_van}-{self.para_tot}"
        )
        return f"{self.ecli}, sectie {self.sectie_nummer} ({self.sectie_titel}), {paras}"


@dataclass(frozen=True)
class _Unit:
    para: int
    tekst: str


def _split_long(text: str, size: int) -> list[str]:
    """Split one over-long paragraph on sentence boundaries, then on words as a last resort."""
    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_RE.split(text):
        while len(sentence) > size:  # a single monster sentence: cut on word boundaries
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > size and current:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _units(sectie: Sectie, size: int) -> Iterator[_Unit]:
    for para in sectie.paragrafen:
        text = para.tekst.strip()
        if not text:
            continue
        if len(text) <= size:
            yield _Unit(para.nummer, text)
        else:
            for piece in _split_long(text, size):
                yield _Unit(para.nummer, piece)


def _tail_for_overlap(units: list[_Unit], overlap: int) -> list[_Unit]:
    """Trailing units of the previous chunk that fit in the overlap budget (never all of them)."""
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units[1:]):
        if total + len(unit.tekst) > overlap:
            break
        tail.insert(0, unit)
        total += len(unit.tekst) + 1
    return tail


def chunk_sectie(sectie: Sectie, config: ChunkConfig) -> Iterator[list[_Unit]]:
    current: list[_Unit] = []
    length = 0
    for unit in _units(sectie, config.size):
        if current and length + 1 + len(unit.tekst) > config.size:
            yield current
            current = _tail_for_overlap(current, config.overlap)
            length = sum(len(u.tekst) + 1 for u in current)
        current.append(unit)
        length += len(unit.tekst) + 1
    if current:
        yield current


def chunk(uitspraak: Uitspraak, config: ChunkConfig | None = None) -> list[Chunk]:
    config = config or ChunkConfig()
    out: list[Chunk] = []
    for sectie in uitspraak.secties:
        for units in chunk_sectie(sectie, config):
            tekst = "\n".join(u.tekst for u in units)
            if len(tekst) < config.min_chars:
                continue
            positie = len(out)
            out.append(
                Chunk(
                    chunk_id=f"{uitspraak.ecli}#{positie}",
                    ecli=uitspraak.ecli,
                    positie=positie,
                    sectie_nummer=sectie.nummer,
                    sectie_titel=sectie.titel,
                    para_van=units[0].para,
                    para_tot=units[-1].para,
                    tekst=tekst,
                )
            )
    return out
