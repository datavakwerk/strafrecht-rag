from itertools import pairwise

import pytest

from strafrecht_rag.index.chunker import ChunkConfig, chunk
from tests.conftest import maak_uitspraak


def _paras(n: int, length: int = 100, prefix: str = "p") -> list[str]:
    return [f"{prefix}{i:02d} " + ("woord " * (length // 6)).strip() for i in range(n)]


def test_config_validation():
    with pytest.raises(ValueError):
        ChunkConfig(size=100, overlap=100)
    with pytest.raises(ValueError):
        ChunkConfig(size=0)


def test_chunks_are_traceable_and_within_size():
    u = maak_uitspraak("ECLI:NL:X:2024:1", {"A": _paras(12, 100)})
    chunks = chunk(u, ChunkConfig(size=350, overlap=100))
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.positie == i
        assert c.chunk_id == f"ECLI:NL:X:2024:1#{i}"
        assert c.sectie_nummer == "1" and c.sectie_titel == "A"
        assert 1 <= c.para_van <= c.para_tot <= 12
        assert len(c.tekst) <= 350
        # every line of the chunk is verbatim a paragraph in the declared range
        source = {p.tekst for p in u.secties[0].paragrafen[c.para_van - 1 : c.para_tot]}
        assert set(c.tekst.split("\n")) <= source


def test_overlap_carries_previous_paragraphs():
    u = maak_uitspraak("ECLI:NL:X:2024:1", {"A": _paras(10, 100)})
    chunks = chunk(u, ChunkConfig(size=350, overlap=120))
    for prev, nxt in pairwise(chunks):
        assert nxt.para_van <= prev.para_tot  # overlapping paragraph range
        assert nxt.para_van > prev.para_van  # but real progress
        shared = set(prev.tekst.split("\n")) & set(nxt.tekst.split("\n"))
        assert shared and sum(len(s) for s in shared) <= 120


def test_no_overlap_when_configured_zero():
    u = maak_uitspraak("ECLI:NL:X:2024:1", {"A": _paras(10, 100)})
    chunks = chunk(u, ChunkConfig(size=350, overlap=0))
    for prev, nxt in pairwise(chunks):
        assert nxt.para_van == prev.para_tot + 1


def test_chunks_never_cross_section_boundaries():
    u = maak_uitspraak("ECLI:NL:X:2024:1", {"A": _paras(4, 100, "a"), "B": _paras(4, 100, "b")})
    chunks = chunk(u, ChunkConfig(size=350, overlap=150))
    assert {c.sectie_titel for c in chunks} == {"A", "B"}
    for c in chunks:
        assert all(
            line.startswith("a" if c.sectie_titel == "A" else "b") for line in c.tekst.split("\n")
        )
    first_b = next(c for c in chunks if c.sectie_titel == "B")
    assert first_b.para_van == 1  # section B starts fresh, no overlap carried across


def test_short_section_is_one_chunk_and_empty_section_none():
    u = maak_uitspraak(
        "ECLI:NL:X:2024:1",
        {"Kort": ["Eén zin van een sectie."], "Leeg": ["", "   "], "Crumb": ["ja"]},
    )
    chunks = chunk(u, ChunkConfig(size=900, overlap=150))
    assert [(c.sectie_titel, c.para_van, c.para_tot) for c in chunks] == [("Kort", 1, 1)]


def test_overlong_paragraph_is_split_on_sentences_and_keeps_para_number():
    long = " ".join(f"Zin nummer {i} over verduistering." for i in range(40))
    u = maak_uitspraak("ECLI:NL:X:2024:1", {"A": ["intro", long, "slot"]})
    chunks = chunk(u, ChunkConfig(size=300, overlap=50))
    middle = [c for c in chunks if c.para_van <= 2 <= c.para_tot]
    assert len(middle) > 2
    assert all(len(c.tekst) <= 300 for c in chunks)
    joined = " ".join(c.tekst for c in chunks)
    assert "Zin nummer 39" in joined and "Zin nummer 0" in joined


def test_monster_sentence_without_punctuation_is_cut_on_words():
    u = maak_uitspraak("ECLI:NL:X:2024:1", {"A": ["woord " * 200]})
    chunks = chunk(u, ChunkConfig(size=200, overlap=20))
    assert chunks and all(len(c.tekst) <= 200 for c in chunks)
    assert all(not c.tekst.startswith(" ") for c in chunks)


def test_deterministic():
    u = maak_uitspraak("ECLI:NL:X:2024:1", {"A": _paras(20, 80)})
    assert chunk(u) == chunk(u)


def test_real_ruling_round_trips_to_source(echte_uitspraak):
    chunks = chunk(echte_uitspraak, ChunkConfig(size=900, overlap=150))
    assert len(chunks) > 10
    by_nr = {s.nummer: s for s in echte_uitspraak.secties}
    for c in chunks:
        sectie = by_nr[c.sectie_nummer]
        paras = sectie.paragrafen[c.para_van - 1 : c.para_tot]
        source_text = " ".join(p.tekst for p in paras)
        for line in c.tekst.split("\n"):
            assert line in source_text  # every chunk line is found verbatim in its source range
    assert chunks[0].sectie_titel == "Inhoudsindicatie"
    assert chunks[0].bron == "ECLI:NL:RBNNE:2024:674, sectie 0 (Inhoudsindicatie), par. 1"
