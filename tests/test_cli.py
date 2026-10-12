import json
from datetime import date

from typer.testing import CliRunner

from strafrecht_rag import cli
from strafrecht_rag.index.store import Filters
from strafrecht_rag.query.answer import Antwoord, GeenBron

runner = CliRunner()


def test_help_lists_the_chain():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "index", "ask", "eval", "demo", "stats"):
        assert command in result.output


def test_ask_routes_filters_and_prints_json(monkeypatch):
    captured = {}

    class FakeGenerator:
        def answer(self, vraag, *, k=None, filters=None):
            captured.update(vraag=vraag, k=k, filters=filters)
            return GeenBron(vraag, "test", 0.1, 0.35)

    monkeypatch.setattr(
        "strafrecht_rag.runtime.generator_for",
        lambda s, llm=None, hybrid=True: (FakeGenerator(), None),
    )
    result = runner.invoke(
        cli.app,
        ["ask", "witwassen", "--json", "--k", "3", "--type", "Gerechtshof", "--van", "2024-01-01"],
    )
    assert result.exit_code == 0, result.output
    assert captured["vraag"] == "witwassen" and captured["k"] == 3
    assert captured["filters"] == Filters(instantie_type="Gerechtshof", datum_van=date(2024, 1, 1))
    payload = json.loads(result.output)
    assert payload["status"] == "geen_bron"
    assert "Geen juridisch advies" in payload["disclaimer"]


def test_ask_renders_answer_without_animation(monkeypatch, store, embedder):
    from strafrecht_rag.query.generator import Generator
    from strafrecht_rag.query.llm import ExtractiveLLM
    from strafrecht_rag.query.retriever import Retriever

    gen = Generator(Retriever(store, embedder, k=3, threshold=0.05), ExtractiveLLM())
    monkeypatch.setattr(
        "strafrecht_rag.runtime.generator_for", lambda s, llm=None, hybrid=True: (gen, store)
    )
    result = runner.invoke(cli.app, ["ask", "verduistering penningmeester", "--no-animate"])
    assert result.exit_code == 0, result.output
    assert "ECLI:NL:RBAMS:2024:1" in result.output
    assert "geverifieerd" in result.output
    assert "Geen juridisch advies" in result.output


def test_ask_renders_no_source_path(monkeypatch, store, embedder):
    from strafrecht_rag.query.generator import Generator
    from strafrecht_rag.query.llm import FakeLLM
    from strafrecht_rag.query.retriever import Retriever

    llm = FakeLLM("nooit aangeroepen [1]")
    gen = Generator(Retriever(store, embedder, k=3, threshold=0.99), llm)
    monkeypatch.setattr(
        "strafrecht_rag.runtime.generator_for", lambda s, llm=None, hybrid=True: (gen, store)
    )
    result = runner.invoke(cli.app, ["ask", "kwantummechanica", "--no-animate"])
    assert result.exit_code == 0, result.output
    assert "Geen bron gevonden" in result.output
    assert llm.calls == []


def test_eval_json_with_extractive_llm(monkeypatch, store, embedder, tmp_path):
    from strafrecht_rag.query.generator import Generator
    from strafrecht_rag.query.llm import ExtractiveLLM
    from strafrecht_rag.query.retriever import Retriever

    gen = Generator(Retriever(store, embedder, k=3, threshold=0.2), ExtractiveLLM())
    monkeypatch.setattr(
        "strafrecht_rag.runtime.generator_for", lambda s, llm=None, hybrid=True: (gen, store)
    )
    questions = tmp_path / "q.json"
    questions.write_text(
        json.dumps(
            {
                "vragen": [
                    {
                        "id": "a",
                        "vraag": "verduistering penningmeester",
                        "verwacht": ["ECLI:NL:RBAMS:2024:1"],
                    },
                    {"id": "b", "vraag": "kwantummechanica", "verwacht": [], "geen_bron": True},
                ]
            }
        )
    )
    result = runner.invoke(cli.app, ["eval", "--json", "--path", str(questions)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["vragen"] == 2 and payload["recall_at_k"] == 1.0
    assert payload["geen_bron_correctheid"] == 1.0
    assert isinstance(payload["per_vraag"], list)


def test_answer_to_dict_roundtrips_json(store, embedder):
    from strafrecht_rag.query.generator import Generator
    from strafrecht_rag.query.llm import ExtractiveLLM
    from strafrecht_rag.query.retriever import Retriever

    gen = Generator(Retriever(store, embedder, k=3, threshold=0.05), ExtractiveLLM())
    result = gen.answer("verduistering penningmeester")
    assert isinstance(result, Antwoord)
    json.dumps(result.to_dict(), ensure_ascii=False)


def test_ask_reports_llm_failure_without_traceback(monkeypatch, store, embedder):
    from strafrecht_rag.query.generator import Generator
    from strafrecht_rag.query.retriever import Retriever

    class BrokenLLM:
        name = "broken"

        def complete(self, system, user):
            raise TimeoutError("timed out")

    gen = Generator(Retriever(store, embedder, k=3, threshold=0.05), BrokenLLM())
    monkeypatch.setattr(
        "strafrecht_rag.runtime.generator_for", lambda s, llm=None, hybrid=True: (gen, store)
    )
    result = runner.invoke(cli.app, ["ask", "verduistering penningmeester", "--no-animate"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "gaf geen antwoord" in result.output and "TimeoutError" in result.output
