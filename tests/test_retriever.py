from datetime import date

from strafrecht_rag.index.store import Filters
from strafrecht_rag.query.retriever import Retriever, _rrf


def test_retrieval_returns_relevant_hits_with_sources(store, embedder):
    r = Retriever(store, embedder, k=4, threshold=0.05)
    result = r.retrieve("verduistering door een penningmeester van een vereniging")
    assert result.boven_drempel
    assert result.hits[0].ecli == "ECLI:NL:RBAMS:2024:1"
    assert result.hits[0].sectie_titel == "Bewezenverklaring"
    assert result.beste_score == result.hits[0].score or result.beste_score >= result.hits[0].score
    assert len(result.hits) <= 4
    assert result.kandidaten >= len(result.hits)


def test_threshold_yields_empty_result(store, embedder):
    r = Retriever(store, embedder, k=4, threshold=0.99)
    result = r.retrieve("kwantummechanica en zwarte gaten")
    assert not result.boven_drempel
    assert result.hits == []
    assert result.beste_score < 0.99
    assert result.drempel == 0.99


def test_filters_are_applied(store, embedder):
    r = Retriever(store, embedder, k=4, threshold=0.05)
    result = r.retrieve(
        "verduisterd gelden",
        filters=Filters(instantie_type="Gerechtshof", datum_van=date(2023, 1, 1)),
    )
    assert {h.ecli for h in result.hits} == {"ECLI:NL:GHARL:2023:7"}
    assert result.filters.instantie_type == "Gerechtshof"


def test_per_ecli_cap_gives_breadth(store, embedder):
    r = Retriever(store, embedder, k=6, threshold=0.0, per_ecli=1)
    result = r.retrieve("gevangenisstraf taakstraf strafmaat")
    eclis = [h.ecli for h in result.hits]
    assert len(eclis) == len(set(eclis))


def test_hybrid_keyword_hit_gets_boosted(store, embedder):
    dense_only = Retriever(store, embedder, k=3, threshold=0.0, hybrid=False)
    hybrid = Retriever(store, embedder, k=3, threshold=0.0, hybrid=True)
    q = "steekpartij horeca"
    assert hybrid.retrieve(q).hits[0].ecli == "ECLI:NL:RBROT:2025:3"
    assert hybrid.retrieve(q).hits[0].bm25_rank == 1
    assert dense_only.retrieve(q).hits[0].bm25_rank is None


def test_rrf_merges_and_marks_keyword_rank(store, embedder):
    q = embedder.embed(["verduisterd"], kind="query")[0]
    dense = store.search_vector(q, 3)
    sparse = store.search_text("verduisterd", q, 3)
    fused = _rrf(dense, sparse)
    assert len({h.chunk_id for h in fused}) == len(fused)
    both = [h for h in fused if h.bm25_rank is not None]
    assert both and both[0].chunk_id in {h.chunk_id for h in sparse}


def test_reranker_sets_relevance_and_drives_threshold(store, embedder):
    from strafrecht_rag.index.reranker import FakeReranker

    r = Retriever(store, embedder, k=3, threshold=0.5, reranker=FakeReranker(), rerank_top=5)
    stages = []
    result = r.retrieve("penningmeester verduisterd clubrekening", on_stage=stages.append)
    assert stages == ["embed", "search", "rerank"]
    assert result.score_soort == "rerank" and result.herordend <= 5
    assert result.hits and all(h.relevantie is not None for h in result.hits)
    assert result.hits[0].ecli == "ECLI:NL:RBAMS:2024:1"
    assert result.hits[0].relevantie == result.beste_score >= 0.5
    rel = [h.relevantie for h in result.hits]
    assert rel == sorted(rel, reverse=True)

    nothing = r.retrieve("kwantummechanica zwarte gaten")
    assert not nothing.boven_drempel and nothing.score_soort == "rerank"
    assert nothing.beste_score < 0.5


def test_without_reranker_threshold_is_on_cosine(store, embedder):
    r = Retriever(store, embedder, k=3, threshold=0.05)
    result = r.retrieve("verduistering")
    assert result.score_soort == "cosine" and result.herordend == 0
    assert all(h.relevantie is None for h in result.hits)


def test_fake_reranker_scores_term_coverage():
    from strafrecht_rag.index.reranker import FakeReranker, make_reranker

    scores = FakeReranker().score(
        "verduistering door een penningmeester", ["de penningmeester", "iets anders", ""]
    )
    assert scores == [0.5, 0.0, 0.0]
    assert make_reranker("none") is None and make_reranker("") is None
    assert isinstance(make_reranker("fake"), FakeReranker)


def test_passages_far_below_threshold_are_not_passed_on(store, embedder):
    from strafrecht_rag.index.reranker import FakeReranker

    r = Retriever(store, embedder, k=6, threshold=0.5, reranker=FakeReranker(), per_ecli=3)
    result = r.retrieve("penningmeester verduisterd clubrekening vereniging")
    assert result.boven_drempel
    assert all((h.relevantie or 0) >= 0.25 for h in result.hits)
    assert len(result.hits) < 6  # the unrelated rulings score 0 and are left out
