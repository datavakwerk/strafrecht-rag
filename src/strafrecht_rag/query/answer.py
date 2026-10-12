"""Answer model: claims with verified references, or an explicit "no source"."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from strafrecht_rag.config import DISCLAIMER
from strafrecht_rag.index.store import Hit


@dataclass(frozen=True)
class Verwijzing:
    nummer: int  # passage number as shown to the model, 1-based
    ecli: str
    sectie_nummer: str
    sectie_titel: str
    para_van: int
    para_tot: int
    chunk_id: str

    @property
    def label(self) -> str:
        paras = (
            f"par. {self.para_van}"
            if self.para_van == self.para_tot
            else f"par. {self.para_van}-{self.para_tot}"
        )
        return f"{self.ecli}, §{self.sectie_nummer} {self.sectie_titel}, {paras}"

    @property
    def url(self) -> str:
        return f"https://uitspraken.rechtspraak.nl/details?id={self.ecli}"

    @classmethod
    def from_hit(cls, nummer: int, hit: Hit) -> Verwijzing:
        return cls(
            nummer=nummer,
            ecli=hit.ecli,
            sectie_nummer=hit.sectie_nummer,
            sectie_titel=hit.sectie_titel,
            para_van=hit.para_van,
            para_tot=hit.para_tot,
            chunk_id=hit.chunk_id,
        )


@dataclass(frozen=True)
class Bewering:
    tekst: str
    verwijzingen: list[Verwijzing]


@dataclass(frozen=True)
class Afgekeurd:
    tekst: str
    reden: str


@dataclass(frozen=True)
class Antwoord:
    vraag: str
    beweringen: list[Bewering]
    afgekeurd: list[Afgekeurd]
    passages: list[Hit]
    llm: str
    ruw: str = ""  # the model's raw output, kept for the eval and for transparency
    duplicaten: int = 0  # repeated claims collapsed by verification
    disclaimer: str = DISCLAIMER

    @property
    def status(self) -> str:
        return "ok" if self.beweringen else "afgekeurd"

    @property
    def eclis(self) -> list[str]:
        seen: dict[str, None] = {}
        for b in self.beweringen:
            for v in b.verwijzingen:
                seen.setdefault(v.ecli, None)
        return list(seen)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "vraag": self.vraag,
            "beweringen": [
                {"tekst": b.tekst, "verwijzingen": [asdict(v) for v in b.verwijzingen]}
                for b in self.beweringen
            ],
            "afgekeurd": [asdict(a) for a in self.afgekeurd],
            "duplicaten": self.duplicaten,
            "passages": [
                {k: v for k, v in asdict(p).items() if k != "rechtsgebieden"} for p in self.passages
            ],
            "llm": self.llm,
            "disclaimer": self.disclaimer,
        }


@dataclass(frozen=True)
class GeenBron:
    vraag: str
    reden: str
    beste_score: float
    drempel: float
    kandidaten: list[Hit] = field(default_factory=list)  # what came closest, for transparency
    disclaimer: str = DISCLAIMER

    status = "geen_bron"

    @property
    def tekst(self) -> str:
        return (
            "Geen bron gevonden. Het corpus bevat geen passage die deze vraag beantwoordt; "
            "er wordt daarom geen antwoord gegenereerd."
        )

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "vraag": self.vraag,
            "reden": self.reden,
            "beste_score": round(self.beste_score, 4),
            "drempel": self.drempel,
            "tekst": self.tekst,
            "disclaimer": self.disclaimer,
        }
