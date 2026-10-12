"""LLM behind one interface: ``complete(system, user) -> str``.

* ``FakeLLM`` — scripted output for tests (including deliberately invented ECLIs).
* ``ExtractiveLLM`` — no model at all: quotes the top passages verbatim with references.
  Runs the whole chain, eval included, on a machine without Ollama or an API key.
* ``OllamaLLM`` — local model via Ollama's chat endpoint (default backend).
* ``AnthropicLLM`` — Claude via the official SDK, when ``ANTHROPIC_API_KEY`` is set.
"""

from __future__ import annotations

import re
from typing import Protocol

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class FakeLLM:
    name = "fake"

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.output


class ExtractiveLLM:
    """Deterministic stand-in: first sentences of the best passages, each cited."""

    name = "extractive"

    def __init__(self, max_passages: int = 3, sentences: int = 2) -> None:
        self.max_passages = max_passages
        self.sentences = sentences

    def complete(self, system: str, user: str) -> str:
        blocks = re.findall(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\n\n---|\Z)", user, re.S)
        lines = []
        for nummer, body in blocks[: self.max_passages]:
            text = " ".join(body.split())
            sentences = SENTENCE_RE.split(text)
            if len(sentences) > 1 and not sentences[0][:1].isupper():
                sentences = sentences[1:]  # the chunk starts mid-sentence; skip the fragment
            snippet = " ".join(sentences[: self.sentences]).strip()
            if snippet:
                lines.append(f"{snippet} [{nummer}]")
        return "\n".join(lines) if lines else "GEEN_BRON"


class OllamaLLM:
    def __init__(
        self, model: str, url: str = "http://localhost:11434", *, timeout: float = 600.0
    ) -> None:
        import httpx

        self.name = f"ollama/{model}"
        self._model = model
        self._client = httpx.Client(base_url=url, timeout=timeout)

    def complete(self, system: str, user: str) -> str:
        response = self._client.post(
            "/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 8192,
                    "num_predict": 400,
                    "repeat_penalty": 1.15,
                },
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class AnthropicLLM:
    def __init__(self, model: str = "claude-opus-5") -> None:
        import anthropic

        self.name = f"anthropic/{model}"
        self._model = model
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            return "GEEN_BRON"
        return "".join(block.text for block in response.content if block.type == "text")


def make_llm(
    backend: str,
    *,
    ollama_model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    anthropic_model: str = "claude-opus-5",
) -> LLM:
    match backend:
        case "extractive":
            return ExtractiveLLM()
        case "ollama":
            return OllamaLLM(ollama_model, ollama_url)
        case "anthropic":
            return AnthropicLLM(anthropic_model)
        case "fake":
            return FakeLLM("GEEN_BRON")
    raise ValueError(f"unknown LLM backend {backend!r}")
