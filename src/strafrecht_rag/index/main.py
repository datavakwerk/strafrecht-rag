"""Index run: data/raw → parse → data/parsed → chunk → embed → store. Idempotent."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from strafrecht_rag.index.chunker import ChunkConfig, chunk
from strafrecht_rag.index.embeddings import Embedder
from strafrecht_rag.index.store import VectorStore
from strafrecht_rag.ingest.model import Uitspraak
from strafrecht_rag.ingest.parser import PARSER_VERSION, ParseError, parse_file


@dataclass(frozen=True)
class IndexEvent:
    ecli: str
    status: str  # "indexed" | "skipped" | "parse_error"
    chunks: int = 0
    detail: str = ""


def embed_text(uitspraak: Uitspraak, sectie_titel: str, tekst: str) -> str:
    """What actually gets embedded: a little context in front of the chunk text."""
    return f"{uitspraak.instantie}, {sectie_titel}\n{tekst}"


def fingerprint(payload: bytes, embedder: Embedder, config: ChunkConfig) -> str:
    """Identity of one indexed document: raw bytes + everything that shapes its chunks."""
    h = hashlib.sha256(payload)
    h.update(f"|{PARSER_VERSION}|{embedder.name}|{config.size}|{config.overlap}".encode())
    return h.hexdigest()


def index_corpus(
    raw_dir: Path,
    parsed_dir: Path,
    store: VectorStore,
    embedder: Embedder,
    config: ChunkConfig,
    *,
    limit: int | None = None,
    force: bool = False,
) -> Iterator[IndexEvent]:
    """Index every XML in ``raw_dir``. Unchanged documents are skipped; re-runs are cheap."""
    paths = sorted(raw_dir.glob("*.xml"))[:limit]
    for path in paths:
        payload = path.read_bytes()
        parsed = parse_file(path)
        if isinstance(parsed, ParseError):
            yield IndexEvent(parsed.bron, "parse_error", detail=parsed.fout)
            continue
        fp = fingerprint(payload, embedder, config)
        if not force and store.doc_fingerprint(parsed.ecli) == fp:
            yield IndexEvent(parsed.ecli, "skipped")
            continue
        parsed.save(parsed_dir)
        chunks = chunk(parsed, config)
        if chunks:
            vectors = embedder.embed(
                [embed_text(parsed, c.sectie_titel, c.tekst) for c in chunks], kind="document"
            )
            store.upsert(parsed, chunks, vectors)
        store.set_doc(parsed.ecli, fp, len(chunks))
        yield IndexEvent(parsed.ecli, "indexed", chunks=len(chunks))
    store.set_meta("embedder", embedder.name)
    store.set_meta("dim", str(embedder.dim))
    store.set_meta("chunk_size", str(config.size))
    store.set_meta("chunk_overlap", str(config.overlap))
