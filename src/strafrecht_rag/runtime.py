"""Wire settings to concrete components. The only place backends are chosen."""

from __future__ import annotations

from strafrecht_rag.config import Settings
from strafrecht_rag.index.chunker import ChunkConfig
from strafrecht_rag.index.embeddings import Embedder, make_embedder
from strafrecht_rag.index.reranker import make_reranker
from strafrecht_rag.index.store import VectorStore
from strafrecht_rag.query.generator import Generator
from strafrecht_rag.query.llm import LLM, make_llm
from strafrecht_rag.query.retriever import Retriever


def open_store(s: Settings) -> VectorStore:
    return VectorStore(s.index_dir / "index.sqlite")


def embedder_for(s: Settings, store: VectorStore | None = None) -> Embedder:
    """Use the embedder the index was built with, unless the index is empty."""
    backend, model = s.embedder, s.embed_model
    recorded = store.get_meta("embedder") if store is not None else None
    if recorded and recorded != "fake" and backend != "fake":
        if recorded.startswith("ollama/"):
            backend, model = "ollama", recorded.split("/", 1)[1]
        else:
            backend, model = "fastembed", recorded
    return make_embedder(backend, model, ollama_url=s.ollama_url)


def chunk_config(s: Settings) -> ChunkConfig:
    return ChunkConfig(size=s.chunk_size, overlap=s.chunk_overlap)


def llm_for(s: Settings, override: str | None = None) -> LLM:
    return make_llm(
        override or s.llm,
        ollama_model=s.ollama_model,
        ollama_url=s.ollama_url,
        anthropic_model=s.anthropic_model,
    )


def generator_for(
    s: Settings, *, llm: str | None = None, hybrid: bool = True, rerank: bool = True
) -> tuple[Generator, VectorStore]:
    store = open_store(s)
    embedder = embedder_for(s, store)
    retriever = Retriever(
        store,
        embedder,
        k=s.top_k,
        threshold=s.score_threshold,
        hybrid=hybrid,
        reranker=make_reranker(s.reranker) if rerank else None,
        rerank_top=s.rerank_top,
    )
    return Generator(retriever, llm_for(s, llm)), store
