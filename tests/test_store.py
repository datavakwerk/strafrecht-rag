from datetime import date

import numpy as np

from strafrecht_rag.index.chunker import ChunkConfig, chunk
from strafrecht_rag.index.embeddings import FakeEmbedder
from strafrecht_rag.index.store import Filters, VectorStore
from tests.conftest import CORPUS, maak_uitspraak


def test_fake_embedder_is_deterministic_and_similarity_aware():
    e = FakeEmbedder()
    a = e.embed(["verduistering door een penningmeester"])
    b = e.embed(["verduistering door een penningmeester"])
    c = e.embed(["steekpartij in de horeca"])
    assert np.allclose(a, b)
    assert (a @ b.T).item() > (a @ c.T).item()
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_upsert_is_idempotent(store, embedder):
    before = store.count()
    u = CORPUS[0]
    chunks = chunk(u, ChunkConfig(size=400, overlap=60))
    store.upsert(u, chunks, embedder.embed([c.tekst for c in chunks]))
    store.upsert(u, chunks, embedder.embed([c.tekst for c in chunks]))
    assert store.count() == before
    assert len(store.eclis()) == 3


def test_reindexing_a_shorter_version_drops_stale_chunks(store, embedder):
    u = CORPUS[0]
    shorter = maak_uitspraak(u.ecli, {"Bewezenverklaring": ["Alleen nog deze ene zin."]})
    chunks = chunk(shorter)
    store.upsert(shorter, chunks, embedder.embed([c.tekst for c in chunks]))
    assert [
        h.chunk_id for h in store.search_vector(embedder.embed(["zin"])[0], 50) if h.ecli == u.ecli
    ] == [f"{u.ecli}#0"]


def test_vector_search_ranks_relevant_chunk_first(store, embedder):
    q = embedder.embed(["verduisterd geld penningmeester vereniging"], kind="query")[0]
    hits = store.search_vector(q, 3)
    assert hits[0].ecli == "ECLI:NL:RBAMS:2024:1"
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert hits[0].sectie_titel == "Bewezenverklaring"
    assert hits[0].para_van == 1
    assert "Bewezenverklaring" in hits[0].bron
    assert hits[0].url.endswith(hits[0].ecli)


def test_metadata_filters(store, embedder):
    q = embedder.embed(["verduisterd gelden"], kind="query")[0]
    hof = store.search_vector(q, 10, Filters(instantie_type="Gerechtshof"))
    assert {h.ecli for h in hof} == {"ECLI:NL:GHARL:2023:7"}
    recent = store.search_vector(q, 10, Filters(datum_van=date(2024, 1, 1)))
    assert {h.datum[:4] for h in recent} == {"2024", "2025"}
    window = store.search_vector(
        q, 10, Filters(datum_van=date(2023, 1, 1), datum_tot=date(2023, 12, 31))
    )
    assert {h.ecli for h in window} == {"ECLI:NL:GHARL:2023:7"}
    assert store.search_vector(q, 10, Filters(instantie="Utrecht")) == []
    assert Filters(instantie_type="Rechtbank", datum_van=date(2024, 1, 1)).describe() == (
        "instantie: Rechtbank, datum 2024-01-01 – …"
    )


def test_fulltext_search_finds_keyword_and_reports_cosine(store, embedder):
    q = embedder.embed(["steekpartij"], kind="query")[0]
    hits = store.search_text("steekpartij horeca", q, 5)
    assert hits and hits[0].ecli == "ECLI:NL:RBROT:2025:3"
    assert hits[0].bm25_rank == 1
    assert -1.0 <= hits[0].score <= 1.0
    assert store.search_text("", q, 5) == []
    assert store.search_text("zzzzzz", q, 5) == []


def test_stats_and_meta(store):
    s = store.stats()
    assert s["uitspraken"] == 3 and s["chunks"] == store.count()
    assert s["per_type"] == {"Rechtbank": 2, "Gerechtshof": 1}
    assert s["per_jaar"] == {"2023": 1, "2024": 1, "2025": 1}
    store.set_meta("embedder", "fake")
    assert store.get_meta("embedder") == "fake"
    assert store.get_meta("missing") is None


def test_empty_store_returns_nothing(tmp_path, embedder):
    empty = VectorStore(tmp_path / "empty.sqlite")
    assert empty.search_vector(embedder.embed(["x"])[0], 5) == []
    assert empty.stats()["uitspraken"] == 0


def test_index_run_is_incremental(tmp_path, embedder):
    from pathlib import Path

    from strafrecht_rag.index.chunker import ChunkConfig
    from strafrecht_rag.index.main import index_corpus

    raw = tmp_path / "raw"
    raw.mkdir()
    fixtures = Path(__file__).parent / "fixtures"
    for name in ("ecli_rbnne_2024_674.xml", "ecli_gharl_2024_2.xml"):
        (raw / name).write_bytes((fixtures / name).read_bytes())
    (raw / "broken.xml").write_bytes(b"<open-rechtspraak/>")
    store = VectorStore(tmp_path / "index.sqlite")
    config = ChunkConfig(size=600, overlap=100)

    first = list(index_corpus(raw, tmp_path / "parsed", store, embedder, config))
    assert sorted(e.status for e in first) == ["indexed", "indexed", "parse_error"]
    total = store.count()
    assert total == sum(e.chunks for e in first)
    assert (tmp_path / "parsed" / "ECLI_NL_RBNNE_2024_674.json").exists()

    second = list(index_corpus(raw, tmp_path / "parsed", store, embedder, config))
    assert sorted(e.status for e in second) == ["parse_error", "skipped", "skipped"]
    assert store.count() == total

    # a changed chunk config is a different index: everything is redone, still no duplicates
    third = list(index_corpus(raw, tmp_path / "parsed", store, embedder, ChunkConfig(400, 50)))
    assert sorted(e.status for e in third) == ["indexed", "indexed", "parse_error"]
    assert store.count() == sum(e.chunks for e in third)
    assert len(store.eclis()) == 2
    assert store.get_meta("chunk_size") == "400"


def test_matrix_cache_sidecar_is_used_and_invalidated(store, embedder):
    q = embedder.embed(["verduisterd"], kind="query")[0]
    first = [h.chunk_id for h in store.search_vector(q, 3)]
    assert store._cache_path.exists()
    fresh = VectorStore(store.path)  # new process: loads from the sidecar
    assert [h.chunk_id for h in fresh.search_vector(q, 3)] == first
    u = CORPUS[2]
    chunks = chunk(u, ChunkConfig(size=200, overlap=20))
    fresh.upsert(u, chunks, embedder.embed([c.tekst for c in chunks]))
    stale = VectorStore(store.path)  # row count changed → cache ignored, rebuilt
    assert stale.count() == fresh.count()
    assert len(stale._load_matrix()[0]) == fresh.count()
