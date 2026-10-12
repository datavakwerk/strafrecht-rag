import json

from strafrecht_rag.eval import EvalVraag, evaluate, load_questions
from strafrecht_rag.query.generator import Generator
from strafrecht_rag.query.llm import ExtractiveLLM, FakeLLM
from strafrecht_rag.query.retriever import Retriever

VRAGEN = [
    EvalVraag(
        "q1", "verduistering door een penningmeester", "semantisch", ["ECLI:NL:RBAMS:2024:1"]
    ),
    EvalVraag(
        "q2",
        "verduisterd gelden bestuurder stichting",
        "gefilterd",
        ["ECLI:NL:GHARL:2023:7"],
        filters=EvalVraag.from_dict(
            {"id": "x", "vraag": "x", "filters": {"instantie_type": "Gerechtshof"}}
        ).filters,
    ),
    EvalVraag(
        "q3", "steekpartij in de horeca", "semantisch", ["ECLI:NL:RBROT:2025:3", "ECLI:NL:X:1:1"]
    ),
    EvalVraag("q4", "kwantummechanica zwarte gaten", "geen_bron", [], geen_bron=True),
]


def test_retrieval_metrics_without_llm(store, embedder):
    gen = Generator(Retriever(store, embedder, k=4, threshold=0.2), FakeLLM("x"))
    report = evaluate(gen, VRAGEN, k=4, with_llm=False)
    assert report.llm == "—"
    by_id = {r.vraag.id: r for r in report.resultaten}
    assert by_id["q1"].recall == 1.0 and by_id["q1"].reciprocal_rank == 1.0
    assert by_id["q2"].opgehaald == ["ECLI:NL:GHARL:2023:7"]
    assert by_id["q3"].recall == 0.5  # one of two expected ECLIs is not in the corpus
    assert by_id["q4"].geen_bron_correct is True and by_id["q4"].status == "ok"
    assert report.recall_at_k == (1.0 + 1.0 + 0.5) / 3
    assert report.hit_at_k == 1.0
    assert report.geen_bron_correctheid == 1.0
    assert report.onterecht_geen_bron == 0.0
    assert report.to_dict()["per_vraag"][3]["status"] == "ok"


def test_generation_metrics_with_extractive_llm(store, embedder):
    gen = Generator(Retriever(store, embedder, k=4, threshold=0.2), ExtractiveLLM())
    seen = []
    report = evaluate(gen, VRAGEN, k=4, on_result=seen.append)
    assert len(seen) == 4
    assert report.bronjuistheid == 1.0
    assert report.claim_acceptance == 1.0
    assert all(r.claims_ok > 0 for r in report.answerable)


def test_hallucinating_llm_lowers_claim_acceptance_but_not_source_correctness(store, embedder):
    llm = FakeLLM("Echte bewering [1]\nVerzonnen ECLI:NL:HR:1999:1 [1]\nZonder bron")
    gen = Generator(Retriever(store, embedder, k=4, threshold=0.2), llm)
    report = evaluate(gen, VRAGEN, k=4)
    assert report.claim_acceptance == 1 / 3
    assert report.bronjuistheid == 1.0  # rejected claims never make it into the answer
    assert report.to_dict()["claim_acceptance"] == 0.333


def test_threshold_too_high_shows_up_as_onterecht_geen_bron(store, embedder):
    gen = Generator(Retriever(store, embedder, k=4, threshold=0.99), ExtractiveLLM())
    report = evaluate(gen, VRAGEN, k=4)
    assert report.onterecht_geen_bron == 1.0
    assert report.geen_bron_correctheid == 1.0
    assert {r.status for r in report.answerable} == {"geen bron"}


def test_question_set_file_is_well_formed():
    vragen = load_questions()
    assert len(vragen) >= 12
    ids = [q.id for q in vragen]
    assert len(ids) == len(set(ids))
    assert sum(q.geen_bron for q in vragen) >= 3
    for q in vragen:
        assert q.geen_bron == (not q.verwacht)
        assert q.type in {"semantisch", "gefilterd", "inhoudelijk", "geen_bron"}
        for e in q.verwacht:
            assert e.startswith("ECLI:NL:")
    filtered = [q for q in vragen if q.type == "gefilterd"]
    assert filtered and all(not q.filters.is_empty() for q in filtered)
    json.loads(json.dumps([q.id for q in vragen]))


def test_llm_failure_is_recorded_per_question_not_fatal(store, embedder):
    class Flaky:
        name = "flaky"
        calls = 0

        def complete(self, system, user):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise TimeoutError("timed out")
            return "Bewering [1]"

    gen = Generator(Retriever(store, embedder, k=4, threshold=0.2), Flaky())
    report = evaluate(gen, VRAGEN, k=4)
    statuses = [r.status for r in report.resultaten]
    assert statuses[0] == "llm-fout" and "TimeoutError" in report.resultaten[0].llm_fout
    assert report.resultaten[0].recall == 1.0  # retrieval metrics survive
    assert "ok" in statuses[1:]
    assert report.to_dict()["per_vraag"][0]["llm_fout"].startswith("TimeoutError")
