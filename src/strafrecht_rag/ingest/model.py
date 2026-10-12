"""The data contract everything downstream builds on: one normalized ``Uitspraak``.

A ruling is metadata plus ordered sections; each section is ordered, numbered
paragraphs. Section number + paragraph number is what makes a chunk traceable
back to the exact passage in the source (PRD §4).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

INSTANTIE_TYPES = {
    "HR": "Hoge Raad",
    "PHR": "Parket bij de Hoge Raad",
    "GH": "Gerechtshof",
    "RB": "Rechtbank",
}


@dataclass(frozen=True)
class Paragraaf:
    nummer: int  # 1-based, ordinal within the section
    tekst: str
    label: str = ""  # the document's own numbering ("3.1") when present


@dataclass(frozen=True)
class Sectie:
    nummer: str  # document numbering when present, else ordinal ("1", "2", ...)
    titel: str
    paragrafen: list[Paragraaf] = field(default_factory=list)

    @property
    def tekst(self) -> str:
        return "\n".join(p.tekst for p in self.paragrafen)


@dataclass(frozen=True)
class Uitspraak:
    ecli: str
    instantie: str
    datum: date
    rechtsgebieden: list[str]
    procedures: list[str]
    zaaknummer: str
    soort: str  # "Uitspraak" | "Conclusie"
    wetsverwijzingen: list[str]
    samenvatting: str
    secties: list[Sectie]

    @property
    def instantie_type(self) -> str:
        """Court type derived from the ECLI code: RBAMS → Rechtbank, GHARL → Gerechtshof."""
        code = self.ecli.split(":")[2] if self.ecli.count(":") >= 3 else ""
        for prefix, label in INSTANTIE_TYPES.items():
            if code.startswith(prefix) and (prefix != "HR" or code == "HR"):
                return label
        return "Overig"

    @property
    def aantal_paragrafen(self) -> int:
        return sum(len(s.paragrafen) for s in self.secties)

    @property
    def url(self) -> str:
        return f"https://uitspraken.rechtspraak.nl/details?id={self.ecli}"

    # ------------------------------------------------------------ persistence
    def to_json(self) -> str:
        payload = asdict(self)
        payload["datum"] = self.datum.isoformat()
        return json.dumps(payload, ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> Uitspraak:
        raw = json.loads(text)
        raw["datum"] = date.fromisoformat(raw["datum"])
        raw["secties"] = [
            Sectie(
                nummer=s["nummer"],
                titel=s["titel"],
                paragrafen=[Paragraaf(**p) for p in s["paragrafen"]],
            )
            for s in raw["secties"]
        ]
        return cls(**raw)

    def save(self, parsed_dir: Path) -> Path:
        parsed_dir.mkdir(parents=True, exist_ok=True)
        target = parsed_dir / f"{self.ecli.replace(':', '_')}.json"
        target.write_text(self.to_json(), encoding="utf-8")
        return target


def load_parsed(parsed_dir: Path) -> list[Uitspraak]:
    paths = sorted(parsed_dir.glob("*.json"))
    return [Uitspraak.from_json(p.read_text(encoding="utf-8")) for p in paths]
