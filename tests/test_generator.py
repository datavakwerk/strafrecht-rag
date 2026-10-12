from strafrecht_rag.query.answer import Antwoord, GeenBron
from strafrecht_rag.query.generator import (
    SYSTEM_PROMPT,
    Generator,
    build_prompt,
    verify,
)
from strafrecht_rag.query.llm import ExtractiveLLM, FakeLLM
from strafrecht_rag.query.retriever import Retriever

VRAAG = "Welke uitspraken gaan over verduistering door een penningmeester?"


def _generator(store, embedder, output: str, threshold: float = 0.05) -> Generator:
    return Generator(Retriever(store, embedder, k=4, threshold=threshold), FakeLLM(output))


def test_claims_with_valid_markers_are_kept_with_traceable_references(store, embedder):
    gen = _generator(
        store,
        embedder,
        "De penningmeester verduisterde geld van de clubrekening [1]\n"
        "- Het hof legde zes maanden gevangenisstraf op [2][3]\n",
    )
    result = gen.answer(VRAAG)
    assert isinstance(result, Antwoord)
    assert result.status == "ok"
    assert [b.tekst for b in result.beweringen] == [
        "De penningmeester verduisterde geld van de clubrekening.",
        "Het hof legde zes maanden gevangenisstraf op.",
    ]
    first = result.beweringen[0].verwijzingen[0]
    assert first.nummer == 1 and first.ecli == result.passages[0].ecli
    assert first.label.startswith(first.ecli)
    assert [v.nummer for v in result.beweringen[1].verwijzingen] == [2, 3]
    assert result.afgekeurd == []
    assert "Geen juridisch advies" in result.disclaimer
    assert result.eclis  # unique ECLIs in citation order
    assert result.to_dict()["status"] == "ok"


def test_invented_ecli_is_rejected(store, embedder):
    gen = _generator(
        store,
        embedder,
        "Zie ook ECLI:NL:HR:2019:9999 over verduistering [1]\n"
        "De penningmeester verduisterde geld [1]",
    )
    result = gen.answer(VRAAG)
    assert isinstance(result, Antwoord)
    assert len(result.beweringen) == 1
    assert len(result.afgekeurd) == 1
    assert "ECLI:NL:HR:2019:9999" in result.afgekeurd[0].reden


def test_claim_without_reference_is_rejected(store, embedder):
    gen = _generator(store, embedder, "Dit is een bewering zonder bron.\nMet bron [1]")
    result = gen.answer(VRAAG)
    assert isinstance(result, Antwoord)
    assert [b.tekst for b in result.beweringen] == ["Met bron."]
    assert result.afgekeurd[0].reden == "geen verwijzing naar een passage"


def test_reference_to_passage_that_was_not_supplied_is_rejected(store, embedder):
    gen = _generator(store, embedder, "Bewering met een verzonnen passage [9]")
    result = gen.answer(VRAAG)
    assert isinstance(result, Antwoord)
    assert result.status == "afgekeurd"
    assert result.beweringen == []
    assert "niet-aangeleverde passage [9]" in result.afgekeurd[0].reden


def test_below_threshold_skips_the_llm_entirely(store, embedder):
    llm = FakeLLM("Dit had nooit gegenereerd mogen worden [1]")
    gen = Generator(Retriever(store, embedder, k=4, threshold=0.99), llm)
    result = gen.answer("kwantummechanica en zwarte gaten")
    assert isinstance(result, GeenBron)
    assert llm.calls == []
    assert "niet aangeroepen" in result.reden
    assert "Geen bron gevonden" in result.tekst
    assert result.to_dict()["status"] == "geen_bron"


def test_model_may_declare_no_source(store, embedder):
    gen = _generator(store, embedder, "GEEN_BRON")
    result = gen.answer(VRAAG)
    assert isinstance(result, GeenBron)
    assert result.kandidaten  # the passages it looked at are still reported
    assert "GEEN_BRON" in result.reden


def test_prompt_contains_numbered_passages_and_rules(store, embedder):
    retrieval = Retriever(store, embedder, k=3, threshold=0.0).retrieve(VRAAG)
    prompt = build_prompt(VRAAG, retrieval.hits)
    assert prompt.startswith(f"Vraag: {VRAAG}")
    for n, hit in enumerate(retrieval.hits, 1):
        assert f"[{n}] {hit.ecli}" in prompt
        assert hit.tekst in prompt
    assert "GEEN_BRON" in SYSTEM_PROMPT
    assert "UITSLUITEND" in SYSTEM_PROMPT


def test_verify_handles_marker_variants(store, embedder):
    hits = Retriever(store, embedder, k=3, threshold=0.0).retrieve(VRAAG).hits
    v = verify("1. Eerste [1, 2]\n* Tweede [3];[1]\n3) Derde [1][1]", hits)
    assert [[r.nummer for r in b.verwijzingen] for b in v.beweringen] == [[1, 2], [3, 1], [1]]
    assert v.afgekeurd == []


def test_extractive_llm_cites_the_passages_it_quotes(store, embedder):
    gen = Generator(Retriever(store, embedder, k=3, threshold=0.05), ExtractiveLLM())
    result = gen.answer(VRAAG)
    assert isinstance(result, Antwoord)
    assert result.status == "ok"
    assert result.afgekeurd == []
    for bewering in result.beweringen:
        passage = result.passages[bewering.verwijzingen[0].nummer - 1]
        assert bewering.tekst.rstrip(".")[:40] in passage.tekst


def test_extractive_llm_skips_leading_sentence_fragment():
    prompt = (
        "Vraag: x\n\nPassages:\n\n[1] ECLI:NL:X:2024:1 · Rb · 2024 · §1 A\n"
        "ontbreekt. De verdachte heeft geld witgewassen. Dat is strafbaar.\n\n---\nAntwoord:"
    )
    out = ExtractiveLLM(sentences=1).complete("", prompt)
    assert out == "De verdachte heeft geld witgewassen. [1]"


def test_blank_lines_and_punctuation_are_not_claims(store, embedder):
    hits = Retriever(store, embedder, k=3, threshold=0.0).retrieve(VRAAG).hits
    v = verify("\n.\n- \nEchte bewering [1]\n\n", hits)
    assert [b.tekst for b in v.beweringen] == ["Echte bewering."]
    assert v.afgekeurd == []


def test_generator_retries_once_when_nothing_survives(store, embedder):
    class TwoShotLLM:
        name = "twoshot"

        def __init__(self):
            self.calls = []

        def complete(self, system, user):
            self.calls.append(user)
            return "Zonder verwijzing" if len(self.calls) == 1 else "Met verwijzing [1]"

    llm = TwoShotLLM()
    gen = Generator(Retriever(store, embedder, k=3, threshold=0.05), llm)
    result = gen.answer(VRAAG)
    assert isinstance(result, Antwoord) and result.status == "ok"
    assert len(llm.calls) == 2 and "opnieuw" in llm.calls[1]
    assert [b.tekst for b in result.beweringen] == ["Met verwijzing."]


def test_no_source_reports_nearest_candidates(store, embedder):
    from strafrecht_rag.index.reranker import FakeReranker

    gen = Generator(
        Retriever(store, embedder, k=3, threshold=0.9, reranker=FakeReranker()), FakeLLM("x")
    )
    result = gen.answer("gevangenisstraf kwantumfysica sterrenstelsel")
    assert isinstance(result, GeenBron)
    assert 1 <= len(result.kandidaten) <= 3


def test_repeated_claims_are_collapsed_with_merged_references(store, embedder):
    gen = _generator(
        store,
        embedder,
        "De verdachte verduisterde subsidiegeld [1]\n"
        "De verdachte verduisterde subsidiegeld. [2]\n"
        "de verdachte verduisterde subsidiegeld [1]\n"
        "Het hof legde een gevangenisstraf op [2]",
    )
    result = gen.answer(VRAAG)
    assert isinstance(result, Antwoord)
    assert [b.tekst for b in result.beweringen] == [
        "De verdachte verduisterde subsidiegeld.",
        "Het hof legde een gevangenisstraf op.",
    ]
    assert [v.nummer for v in result.beweringen[0].verwijzingen] == [1, 2]
    assert result.duplicaten == 2
    assert result.to_dict()["duplicaten"] == 2
