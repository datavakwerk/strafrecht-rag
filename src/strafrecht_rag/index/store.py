"""Embedded vector store: SQLite for metadata + FTS5, NumPy for the vectors.

Why not a vector database (PRD OQ2): the corpus is tens of thousands of chunks; an
exact cosine scan over a float32 matrix of that size takes single-digit milliseconds,
and SQLite's FTS5 gives BM25 keyword search for hybrid retrieval for free. One file,
zero services, no platform-specific wheels, idempotent upserts on ``chunk_id``. The
interface (``upsert`` / ``search_vector`` / ``search_text``) is what the rest of the
system depends on, so swapping in an ANN store later is a local change.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from strafrecht_rag.index.chunker import Chunk
from strafrecht_rag.index.stopwords import STOPWORDS_NL
from strafrecht_rag.ingest.model import Uitspraak

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    ecli TEXT NOT NULL,
    instantie TEXT NOT NULL,
    instantie_type TEXT NOT NULL,
    datum TEXT NOT NULL,
    rechtsgebieden TEXT NOT NULL,
    sectie_nummer TEXT NOT NULL,
    sectie_titel TEXT NOT NULL,
    para_van INTEGER NOT NULL,
    para_tot INTEGER NOT NULL,
    positie INTEGER NOT NULL,
    tekst TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_ecli ON chunks(ecli);
CREATE INDEX IF NOT EXISTS chunks_filters ON chunks(instantie_type, datum);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    tekst, content='chunks', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, tekst) VALUES (new.id, new.tekst);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, tekst) VALUES ('delete', old.id, old.tekst);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, tekst) VALUES ('delete', old.id, old.tekst);
    INSERT INTO chunks_fts(rowid, tekst) VALUES (new.id, new.tekst);
END;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS docs (
    ecli TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, chunks INTEGER NOT NULL
);
"""

COLUMNS = (
    "chunk_id, ecli, instantie, instantie_type, datum, rechtsgebieden, sectie_nummer, "
    "sectie_titel, para_van, para_tot, positie, tekst"
)


@dataclass(frozen=True)
class Filters:
    instantie_type: str | None = None  # "Rechtbank" | "Gerechtshof" | "Hoge Raad" | ...
    instantie: str | None = None  # substring, e.g. "Amsterdam"
    datum_van: date | None = None
    datum_tot: date | None = None
    rechtsgebied: str | None = None

    def sql(self) -> tuple[str, list]:
        clauses, params = [], []
        if self.instantie_type:
            clauses.append("instantie_type = ?")
            params.append(self.instantie_type)
        if self.instantie:
            clauses.append("instantie LIKE ?")
            params.append(f"%{self.instantie}%")
        if self.datum_van:
            clauses.append("datum >= ?")
            params.append(self.datum_van.isoformat())
        if self.datum_tot:
            clauses.append("datum <= ?")
            params.append(self.datum_tot.isoformat())
        if self.rechtsgebied:
            clauses.append("rechtsgebieden LIKE ?")
            params.append(f"%{self.rechtsgebied}%")
        return (" AND ".join(clauses) if clauses else "1=1"), params

    def is_empty(self) -> bool:
        return self == Filters()

    def describe(self) -> str:
        parts = []
        if self.instantie_type:
            parts.append(f"instantie: {self.instantie_type}")
        if self.instantie:
            parts.append(f"instantie bevat '{self.instantie}'")
        if self.datum_van or self.datum_tot:
            parts.append(f"datum {self.datum_van or '…'} – {self.datum_tot or '…'}")
        if self.rechtsgebied:
            parts.append(f"rechtsgebied: {self.rechtsgebied}")
        return ", ".join(parts) or "geen filters"


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    ecli: str
    instantie: str
    instantie_type: str
    datum: str
    sectie_nummer: str
    sectie_titel: str
    para_van: int
    para_tot: int
    positie: int
    tekst: str
    score: float  # cosine similarity of the chunk to the query
    bm25_rank: int | None = None  # rank in the keyword search when it also matched there
    relevantie: float | None = None  # cross-encoder relevance probability, when reranked
    rechtsgebieden: list[str] = field(default_factory=list)

    @property
    def bron(self) -> str:
        paras = (
            f"par. {self.para_van}"
            if self.para_van == self.para_tot
            else f"par. {self.para_van}-{self.para_tot}"
        )
        return f"{self.ecli}, §{self.sectie_nummer} {self.sectie_titel}, {paras}"

    @property
    def url(self) -> str:
        return f"https://uitspraken.rechtspraak.nl/details?id={self.ecli}"


def _row_to_hit(row: sqlite3.Row, score: float, bm25_rank: int | None = None) -> Hit:
    return Hit(
        chunk_id=row["chunk_id"],
        ecli=row["ecli"],
        instantie=row["instantie"],
        instantie_type=row["instantie_type"],
        datum=row["datum"],
        sectie_nummer=row["sectie_nummer"],
        sectie_titel=row["sectie_titel"],
        para_van=row["para_van"],
        para_tot=row["para_tot"],
        positie=row["positie"],
        tekst=row["tekst"],
        score=float(score),
        bm25_rank=bm25_rank,
        rechtsgebieden=[r for r in row["rechtsgebieden"].split(";") if r],
    )


class VectorStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # WAL: readers (an `ask` while `index` runs) never block the writer; 30 s busy timeout
        self._db = sqlite3.connect(path, timeout=30.0)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._ids: np.ndarray | None = None
        self._matrix: np.ndarray | None = None

    def close(self) -> None:
        self._db.close()

    # ------------------------------------------------------------------ meta
    def set_meta(self, key: str, value: str) -> None:
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def get_meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    # ------------------------------------------------------------------ docs
    def doc_fingerprint(self, ecli: str) -> str | None:
        row = self._db.execute("SELECT fingerprint FROM docs WHERE ecli = ?", (ecli,)).fetchone()
        return row["fingerprint"] if row else None

    def set_doc(self, ecli: str, fingerprint: str, chunks: int) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO docs(ecli, fingerprint, chunks) VALUES (?, ?, ?)",
                (ecli, fingerprint, chunks),
            )

    # ------------------------------------------------------------------ write
    def upsert(self, uitspraak: Uitspraak, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        rows = [
            (
                c.chunk_id,
                c.ecli,
                uitspraak.instantie,
                uitspraak.instantie_type,
                uitspraak.datum.isoformat(),
                ";".join(uitspraak.rechtsgebieden),
                c.sectie_nummer,
                c.sectie_titel,
                c.para_van,
                c.para_tot,
                c.positie,
                c.tekst,
                np.asarray(v, dtype=np.float32).tobytes(),
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self._db:
            # drop chunks of an earlier, longer version of the same ruling
            keep = [c.chunk_id for c in chunks]
            placeholders = ",".join("?" for _ in keep) or "''"
            self._db.execute(
                f"DELETE FROM chunks WHERE ecli = ? AND chunk_id NOT IN ({placeholders})",
                [uitspraak.ecli, *keep],
            )
            self._db.executemany(
                f"""INSERT INTO chunks ({COLUMNS}, vector)
                    VALUES ({",".join("?" for _ in range(13))})
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        instantie=excluded.instantie, instantie_type=excluded.instantie_type,
                        datum=excluded.datum, rechtsgebieden=excluded.rechtsgebieden,
                        sectie_nummer=excluded.sectie_nummer, sectie_titel=excluded.sectie_titel,
                        para_van=excluded.para_van, para_tot=excluded.para_tot,
                        positie=excluded.positie, tekst=excluded.tekst, vector=excluded.vector""",
                rows,
            )
        self._matrix = None

    # ------------------------------------------------------------------ read
    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def eclis(self) -> list[str]:
        return [r[0] for r in self._db.execute("SELECT DISTINCT ecli FROM chunks ORDER BY ecli")]

    def stats(self) -> dict:
        q = self._db.execute
        return {
            "chunks": self.count(),
            "uitspraken": q("SELECT COUNT(DISTINCT ecli) FROM chunks").fetchone()[0],
            "per_type": dict(
                q(
                    "SELECT instantie_type, COUNT(DISTINCT ecli) FROM chunks "
                    "GROUP BY instantie_type ORDER BY 2 DESC"
                ).fetchall()
            ),
            "per_jaar": dict(
                q(
                    "SELECT substr(datum,1,4), COUNT(DISTINCT ecli) FROM chunks "
                    "GROUP BY 1 ORDER BY 1"
                ).fetchall()
            ),
            "instanties": q("SELECT COUNT(DISTINCT instantie) FROM chunks").fetchone()[0],
        }

    @property
    def _cache_path(self) -> Path:
        return self.path.with_suffix(".vectors.npz")

    def _load_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """All vectors as one matrix; a sidecar .npz makes this sub-second after an index run."""
        if self._matrix is not None:
            return self._ids, self._matrix
        count, max_id = self._db.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks"
        ).fetchone()
        if self._cache_path.exists():
            try:
                cached = np.load(self._cache_path)
                if len(cached["ids"]) == count and (
                    count == 0 or int(cached["ids"].max()) == max_id
                ):
                    self._ids, self._matrix = cached["ids"], cached["matrix"]
                    return self._ids, self._matrix
            except (OSError, ValueError, KeyError):
                pass
        rows = self._db.execute("SELECT id, vector FROM chunks ORDER BY id").fetchall()
        self._ids = np.array([r["id"] for r in rows], dtype=np.int64)
        self._matrix = (
            np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )
        if count:
            tmp = self._cache_path.with_suffix(".tmp.npz")
            np.savez(tmp, ids=self._ids, matrix=self._matrix)
            tmp.replace(self._cache_path)
        return self._ids, self._matrix

    def _candidate_ids(self, filters: Filters | None) -> np.ndarray | None:
        if filters is None or filters.is_empty():
            return None
        where, params = filters.sql()
        rows = self._db.execute(f"SELECT id FROM chunks WHERE {where}", params).fetchall()
        return np.array([r["id"] for r in rows], dtype=np.int64)

    def _fetch(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._db.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids)
        return {r["id"]: r for r in rows}

    def search_vector(
        self, vector: np.ndarray, k: int, filters: Filters | None = None
    ) -> list[Hit]:
        ids, matrix = self._load_matrix()
        if matrix.size == 0:
            return []
        candidates = self._candidate_ids(filters)
        if candidates is not None:
            mask = np.isin(ids, candidates)
            ids, matrix = ids[mask], matrix[mask]
            if matrix.size == 0:
                return []
        scores = matrix @ np.asarray(vector, dtype=np.float32)
        top = np.argsort(-scores)[:k]
        rows = self._fetch([int(ids[i]) for i in top])
        return [_row_to_hit(rows[int(ids[i])], scores[i]) for i in top]

    def search_text(
        self, query: str, vector: np.ndarray, k: int, filters: Filters | None = None
    ) -> list[Hit]:
        """BM25 keyword search (FTS5); ``score`` is the cosine to ``vector`` so hits compare."""
        terms = [
            t for t in re.findall(r"\w+", query.lower()) if len(t) > 2 and t not in STOPWORDS_NL
        ]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        where, params = (filters or Filters()).sql()
        rows = self._db.execute(
            f"""SELECT chunks.* FROM chunks_fts JOIN chunks ON chunks.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ? AND {where} ORDER BY bm25(chunks_fts) LIMIT ?""",
            [match, *params, k],
        ).fetchall()
        q = np.asarray(vector, dtype=np.float32)
        out = []
        for rank, row in enumerate(rows, 1):
            cosine = float(np.frombuffer(row["vector"], dtype=np.float32) @ q)
            out.append(_row_to_hit(row, cosine, bm25_rank=rank))
        return out

    def get_chunk(self, chunk_id: str) -> Hit | None:
        row = self._db.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return _row_to_hit(row, 0.0) if row else None
