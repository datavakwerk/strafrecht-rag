from datetime import date
from pathlib import Path

import httpx
import pytest

from strafrecht_rag.ingest.client import (
    RechtspraakClient,
    ZoekResultaat,
    ingest,
    parse_zoek_feed,
    raw_path,
    zoek_aantal,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _feed(eclis: list[str]) -> bytes:
    entries = "".join(
        f'<entry><id>{e}</id><title type="text">{e}, Hof</title>'
        f'<summary type="text">samenvatting {e}</summary></entry>'
        for e in eclis
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?><feed xmlns="http://www.w3.org/2005/Atom">'
        f'<subtitle type="text">Aantal gevonden ECLI\'s: {len(eclis)}</subtitle>{entries}</feed>'
    ).encode()


def _client(handler, **kwargs) -> RechtspraakClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="https://data.rechtspraak.nl/uitspraken")
    return RechtspraakClient(http, min_interval=0, sleep=lambda _: None, **kwargs)


def test_parse_recorded_feed():
    hits = parse_zoek_feed((FIXTURES / "zoek_hof_2024.xml").read_bytes())
    assert [h.ecli for h in hits] == ["ECLI:NL:GHARL:2024:2", "ECLI:NL:GHARL:2024:3"]
    assert hits[0].titel.startswith("ECLI:NL:GHARL:2024:2, Gerechtshof Arnhem-Leeuwarden")
    assert zoek_aantal((FIXTURES / "zoek_hof_2024.xml").read_bytes()) > 0


def test_zoek_paginates_until_short_page(monkeypatch):
    monkeypatch.setattr("strafrecht_rag.ingest.client.PAGE_SIZE", 2)
    pages = {"0": ["A", "B"], "2": ["C", "D"], "4": ["E"]}
    seen_params = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        seen_params.append(dict(params))
        return httpx.Response(200, content=_feed(pages[params["from"]]))

    hits = list(_client(handler).zoek(date(2024, 3, 1), date(2024, 3, 2)))
    assert [h.ecli for h in hits] == ["A", "B", "C", "D", "E"]
    assert [p["from"] for p in seen_params] == ["0", "2", "4"]
    assert seen_params[0]["return"] == "DOC"
    assert seen_params[0]["subject"].endswith("#strafrecht")


def test_retries_on_server_error_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=b"<xml/>")

    assert _client(handler).haal_document("ECLI:NL:X:2024:1") == b"<xml/>"
    assert calls["n"] == 3


def test_gives_up_after_max_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler, max_retries=2).haal_document("ECLI:NL:X:2024:1")


def test_rate_limit_waits_between_requests():
    waits = []
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"<xml/>"))
    http = httpx.Client(transport=transport, base_url="https://x")
    client = RechtspraakClient(http, min_interval=0.5, sleep=waits.append)
    client.haal_document("A")
    client.haal_document("B")
    assert len(waits) == 1 and 0 < waits[0] <= 0.5


def test_ingest_resumes_without_refetching(tmp_path):
    fetched = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/zoeken"):
            return httpx.Response(200, content=_feed(["ECLI:NL:A:2024:1", "ECLI:NL:A:2024:2"]))
        fetched.append(request.url.params["id"])
        return httpx.Response(200, content=b"<open-rechtspraak/>")

    windows = [(date(2024, 3, 1), date(2024, 3, 2))]
    first = list(ingest(_client(handler), tmp_path, windows))
    assert [e.status for e in first] == ["fetched", "fetched"]
    assert raw_path(tmp_path, "ECLI:NL:A:2024:1").read_bytes() == b"<open-rechtspraak/>"

    second = list(ingest(_client(handler), tmp_path, windows))
    assert [e.status for e in second] == ["skipped", "skipped"]
    assert fetched == ["ECLI:NL:A:2024:1", "ECLI:NL:A:2024:2"]


def test_ingest_records_error_per_document(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/zoeken"):
            return httpx.Response(200, content=_feed(["ECLI:NL:A:2024:1", "ECLI:NL:A:2024:2"]))
        if request.url.params["id"].endswith(":1"):
            return httpx.Response(404)
        return httpx.Response(200, content=b"<open-rechtspraak/>")

    events = list(ingest(_client(handler), tmp_path, [(date(2024, 3, 1), date(2024, 3, 2))]))
    assert [e.status for e in events] == ["error", "fetched"]
    assert "404" in events[0].detail
    assert not raw_path(tmp_path, "ECLI:NL:A:2024:1").exists()


def test_zoekresultaat_is_frozen():
    hit = ZoekResultaat("E", "t", "s")
    with pytest.raises(AttributeError):
        hit.ecli = "X"  # type: ignore[misc]
