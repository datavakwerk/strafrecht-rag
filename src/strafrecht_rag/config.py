"""Runtime configuration: environment / .env with working defaults for everything.

Nothing in here is required to run the chain locally; the defaults reproduce the
numbers in the README. See `.env.example` for the knobs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DISCLAIMER = (
    "Geen juridisch advies: dit is een zoek- en samenvattingshulpmiddel over gepubliceerde, "
    "geanonimiseerde uitspraken. Controleer de bron via de ECLI."
)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _windows(spec: str) -> list[tuple[date, date]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        out.append((date.fromisoformat(a), date.fromisoformat(b)))
    return out


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("RAG_DATA_DIR", "data")))
    date_windows: list[tuple[date, date]] = field(
        default_factory=lambda: _windows(
            _env(
                "RAG_DATE_WINDOWS",
                "2023-03-01:2023-03-15,2024-03-01:2024-03-15,2025-03-01:2025-03-15",
            )
        )
    )
    chunk_size: int = field(default_factory=lambda: int(_env("RAG_CHUNK_SIZE", "560")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("RAG_CHUNK_OVERLAP", "120")))
    top_k: int = field(default_factory=lambda: int(_env("RAG_TOP_K", "6")))
    score_threshold: float = field(
        default_factory=lambda: float(_env("RAG_SCORE_THRESHOLD", "0.35"))
    )
    reranker: str = field(
        default_factory=lambda: _env("RAG_RERANKER", "jinaai/jina-reranker-v2-base-multilingual")
    )
    rerank_top: int = field(default_factory=lambda: int(_env("RAG_RERANK_TOP", "12")))
    embedder: str = field(default_factory=lambda: _env("RAG_EMBEDDER", "fastembed"))
    embed_model: str = field(
        default_factory=lambda: _env(
            "RAG_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    llm: str = field(default_factory=lambda: _env("RAG_LLM", "ollama"))
    ollama_url: str = field(
        default_factory=lambda: _env("RAG_OLLAMA_URL", "http://localhost:11434")
    )
    ollama_model: str = field(default_factory=lambda: _env("RAG_OLLAMA_MODEL", "llama3.2"))
    anthropic_model: str = field(
        default_factory=lambda: _env("RAG_ANTHROPIC_MODEL", "claude-sonnet-5")
    )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


def settings() -> Settings:
    return Settings()
