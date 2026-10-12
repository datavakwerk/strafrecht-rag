"""Thin client for Rechtspraak.nl Open Data.

Two endpoints, no key:

* ``/uitspraken/zoeken`` — Atom feed of ECLIs matching date / creator / subject filters,
  paginated with ``from`` + ``max`` (max ≤ 1000).
* ``/uitspraken/content?id=ECLI`` — the full XML of one ruling.

Politeness is built in, not bolted on: a minimum interval between requests and
exponential backoff on 429 / 5xx / transport errors.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

BASE_URL = "https://data.rechtspraak.nl/uitspraken"
SUBJECT_STRAFRECHT = "http://psi.rechtspraak.nl/rechtsgebied#strafrecht"
ATOM = "{http://www.w3.org/2005/Atom}"
PAGE_SIZE = 1000
RETRY_STATUSES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ZoekResultaat:
    ecli: str
    titel: str
    samenvatting: str


class RechtspraakClient:
    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        min_interval: float = 0.25,
        max_retries: int = 4,
        backoff: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client or httpx.Client(
            base_url=BASE_URL, timeout=30.0, headers={"User-Agent": "strafrecht-rag/0.1"}
        )
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._backoff = backoff
        self._sleep = sleep
        self._last_request = 0.0

    # ------------------------------------------------------------------ transport
    def _get(self, path: str, params: dict | list[tuple[str, str]]) -> bytes:
        attempt = 0
        while True:
            wait = self._min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = time.monotonic()
            try:
                response = self._client.get(path, params=params)
            except httpx.TransportError:
                if attempt >= self._max_retries:
                    raise
                self._sleep(self._backoff * 2**attempt)
                attempt += 1
                continue
            if response.status_code in RETRY_STATUSES and attempt < self._max_retries:
                self._sleep(self._backoff * 2**attempt)
                attempt += 1
                continue
            response.raise_for_status()
            return response.content

    # ------------------------------------------------------------------ search
    def zoek(
        self,
        van: date,
        tot: date,
        *,
        subject: str = SUBJECT_STRAFRECHT,
        creator: str | None = None,
    ) -> Iterator[ZoekResultaat]:
        """Yield every ECLI with a full document in the window, following pagination."""
        offset = 0
        while True:
            params: list[tuple[str, str]] = [
                ("subject", subject),
                ("date", van.isoformat()),
                ("date", tot.isoformat()),
                ("return", "DOC"),
                ("max", str(PAGE_SIZE)),
                ("from", str(offset)),
            ]
            if creator:
                params.append(("creator", creator))
            entries = parse_zoek_feed(self._get("/zoeken", params))
            yield from entries
            if len(entries) < PAGE_SIZE:
                return
            offset += PAGE_SIZE

    def haal_document(self, ecli: str) -> bytes:
        return self._get("/content", {"id": ecli})


def parse_zoek_feed(payload: bytes) -> list[ZoekResultaat]:
    root = ET.fromstring(payload)
    out = []
    for entry in root.iter(f"{ATOM}entry"):
        ecli = (entry.findtext(f"{ATOM}id") or "").strip()
        if not ecli:
            continue
        out.append(
            ZoekResultaat(
                ecli=ecli,
                titel=(entry.findtext(f"{ATOM}title") or "").strip(),
                samenvatting=(entry.findtext(f"{ATOM}summary") or "").strip(),
            )
        )
    return out


def zoek_aantal(payload: bytes) -> int:
    """Total hit count as reported by the feed subtitle ("Aantal gevonden ECLI's: N")."""
    root = ET.fromstring(payload)
    subtitle = root.findtext(f"{ATOM}subtitle") or ""
    digits = "".join(ch for ch in subtitle if ch.isdigit())
    return int(digits) if digits else 0


# ---------------------------------------------------------------------- run
@dataclass(frozen=True)
class IngestEvent:
    ecli: str
    status: str  # "fetched" | "skipped" | "error"
    detail: str = ""


def raw_path(raw_dir: Path, ecli: str) -> Path:
    return raw_dir / f"{ecli.replace(':', '_')}.xml"


def ingest(
    client: RechtspraakClient,
    raw_dir: Path,
    windows: list[tuple[date, date]],
    *,
    limit: int | None = None,
) -> Iterator[IngestEvent]:
    """Fetch every ruling in the windows into ``raw_dir``; already-present files are skipped.

    Restarting resumes without duplicate work (PRD Definition of Done). One failing
    document yields one error event, never a crashed batch.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    seen = 0
    for van, tot in windows:
        for hit in client.zoek(van, tot):
            if limit is not None and seen >= limit:
                return
            seen += 1
            target = raw_path(raw_dir, hit.ecli)
            if target.exists():
                yield IngestEvent(hit.ecli, "skipped")
                continue
            try:
                payload = client.haal_document(hit.ecli)
            except httpx.HTTPError as exc:
                yield IngestEvent(hit.ecli, "error", str(exc))
                continue
            tmp = target.with_suffix(".xml.part")
            tmp.write_bytes(payload)
            tmp.replace(target)
            yield IngestEvent(hit.ecli, "fetched")
