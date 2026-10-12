from datetime import date
from pathlib import Path

import pytest

from strafrecht_rag.index.chunker import ChunkConfig, chunk
from strafrecht_rag.index.embeddings import FakeEmbedder
from strafrecht_rag.index.store import VectorStore
from strafrecht_rag.ingest.model import Paragraaf, Sectie, Uitspraak
from strafrecht_rag.ingest.parser import parse_xml

FIXTURES = Path(__file__).parent / "fixtures"


def maak_uitspraak(
    ecli: str,
    secties: dict[str, list[str]],
    *,
    instantie: str = "Rechtbank Amsterdam",
    datum: date = date(2024, 3, 1),
) -> Uitspraak:
    """Small synthetic ruling: {section title: [paragraph texts]}."""
    return Uitspraak(
        ecli=ecli,
        instantie=instantie,
        datum=datum,
        rechtsgebieden=["Strafrecht"],
        procedures=["Eerste aanleg - meervoudig"],
        zaaknummer="13/000000-24",
        soort="Uitspraak",
        wetsverwijzingen=[],
        samenvatting="",
        secties=[
            Sectie(
                nummer=str(i + 1),
                titel=titel,
                paragrafen=[Paragraaf(nummer=j + 1, tekst=t) for j, t in enumerate(paras)],
            )
            for i, (titel, paras) in enumerate(secties.items())
        ],
    )


CORPUS = [
    maak_uitspraak(
        "ECLI:NL:RBAMS:2024:1",
        {
            "Bewezenverklaring": [
                "De penningmeester van de vereniging heeft geld van de clubrekening verduisterd.",
                "Hij heeft in totaal 40.000 euro overgemaakt naar zijn eigen rekening.",
            ],
            "Strafmaat": [
                "De rechtbank weegt mee dat verdachte het vertrouwen van de leden ernstig heeft "
                "geschonden en legt een taakstraf van 240 uur op."
            ],
        },
        instantie="Rechtbank Amsterdam",
        datum=date(2024, 3, 4),
    ),
    maak_uitspraak(
        "ECLI:NL:GHARL:2023:7",
        {
            "Beoordeling": [
                "Het hof acht bewezen dat verdachte als bestuurder van de stichting gelden heeft "
                "verduisterd die hem uit hoofde van zijn functie waren toevertrouwd.",
            ],
            "Strafmaat": ["Het hof legt een gevangenisstraf van zes maanden op."],
        },
        instantie="Gerechtshof Arnhem-Leeuwarden",
        datum=date(2023, 3, 9),
    ),
    maak_uitspraak(
        "ECLI:NL:RBROT:2025:3",
        {
            "Tenlastelegging": [
                "Verdachte wordt verweten dat hij met een mes heeft gestoken tijdens een ruzie "
                "in de horeca.",
            ],
            "Strafmaat": [
                "De rechtbank legt een gevangenisstraf van drie jaar op voor de steekpartij."
            ],
        },
        instantie="Rechtbank Rotterdam",
        datum=date(2025, 3, 3),
    ),
]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path, embedder) -> VectorStore:
    store = VectorStore(tmp_path / "index.sqlite")
    config = ChunkConfig(size=400, overlap=60)
    for u in CORPUS:
        chunks = chunk(u, config)
        store.upsert(u, chunks, embedder.embed([c.tekst for c in chunks]))
    yield store
    store.close()


@pytest.fixture
def echte_uitspraak() -> Uitspraak:
    return parse_xml((FIXTURES / "ecli_rbnne_2024_674.xml").read_bytes())
