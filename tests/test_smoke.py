import importlib

import pytest

PACKAGES = [
    "strafrecht_rag",
    "strafrecht_rag.config",
    "strafrecht_rag.ingest.client",
    "strafrecht_rag.ingest.parser",
    "strafrecht_rag.ingest.model",
    "strafrecht_rag.index.chunker",
    "strafrecht_rag.index.embeddings",
    "strafrecht_rag.index.reranker",
    "strafrecht_rag.index.store",
    "strafrecht_rag.index.main",
    "strafrecht_rag.query.retriever",
    "strafrecht_rag.query.generator",
    "strafrecht_rag.query.llm",
    "strafrecht_rag.query.answer",
    "strafrecht_rag.eval",
    "strafrecht_rag.runtime",
    "strafrecht_rag.ui",
    "strafrecht_rag.cli",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_imports(name):
    assert importlib.import_module(name)


def test_settings_have_working_defaults(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("RAG_"):
            monkeypatch.delenv(key)
    from strafrecht_rag.config import DISCLAIMER, settings

    s = settings()
    assert s.chunk_overlap < s.chunk_size
    assert 0 < s.score_threshold < 1
    assert len(s.date_windows) == 3
    assert "Geen juridisch advies" in DISCLAIMER
