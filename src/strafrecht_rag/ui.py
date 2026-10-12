"""Terminal rendering for the CLI: stages, retrieval table, answer, no-source card.

Everything visual lives here; ``cli.py`` only wires commands to the chain.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from strafrecht_rag.index.store import Hit
from strafrecht_rag.query.answer import Antwoord, GeenBron
from strafrecht_rag.query.retriever import Retrieval

ACCENT = "#F5B942"  # amber: ECLI, headers
TEAL = "#2DD4BF"  # scores, verified
RED = "#F87171"  # rejected
MUTED = "#8B93A7"
BG_PANEL = "#12151C"

console = Console(highlight=False)
TITLE_MAX = 56


def kort(titel: str, limit: int = TITLE_MAX) -> str:
    """Section titles can be a whole charge; keep the display on one line."""
    return titel if len(titel) <= limit else titel[: limit - 1].rstrip() + "…"


def banner(title: str, subtitle: str = "") -> Panel:
    text = Text()
    text.append("strafrecht", style=f"bold {ACCENT}")
    text.append("-rag", style="bold white")
    text.append("  ·  ", style=MUTED)
    text.append(title, style="bold white")
    if subtitle:
        text.append(f"\n{subtitle}", style=MUTED)
    return Panel(text, box=box.ROUNDED, border_style=ACCENT, padding=(0, 2))


# ------------------------------------------------------------------ stages
@dataclass
class Stage:
    label: str
    status: str = "pending"  # pending | running | done | skipped
    detail: str = ""
    started: float = 0.0
    seconds: float = 0.0


@dataclass
class Stages:
    stages: list[Stage] = field(default_factory=list)
    _spinner: Spinner = field(default_factory=lambda: Spinner("dots", style=ACCENT))

    def add(self, label: str) -> None:
        self.stages.append(Stage(label))

    def start(self, index: int) -> None:
        self.stages[index].status = "running"
        self.stages[index].started = time.perf_counter()

    def done(self, index: int, detail: str = "") -> None:
        s = self.stages[index]
        s.status, s.detail = "done", detail
        s.seconds = time.perf_counter() - s.started if s.started else 0.0

    def skip(self, index: int, detail: str = "") -> None:
        self.stages[index].status, self.stages[index].detail = "skipped", detail

    def fail(self, index: int, detail: str = "") -> None:
        self.stages[index].status, self.stages[index].detail = "failed", detail

    def __rich__(self) -> RenderableType:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=2)
        grid.add_column(min_width=26)
        grid.add_column(style=MUTED)
        for s in self.stages:
            if s.status == "running":
                icon: RenderableType = self._spinner
                label = Text(s.label, style="bold white")
            elif s.status == "done":
                icon = Text("✔", style=f"bold {TEAL}")
                label = Text(s.label, style="white")
            elif s.status == "skipped":
                icon = Text("○", style=f"bold {ACCENT}")
                label = Text(s.label, style=MUTED)
            elif s.status == "failed":
                icon = Text("✘", style=f"bold {RED}")
                label = Text(s.label, style=RED)
            else:
                icon = Text("·", style=MUTED)
                label = Text(s.label, style=MUTED)
            detail = Text(s.detail, style=MUTED)
            if s.status == "done" and s.seconds:
                detail.append(
                    f"  {s.seconds * 1000:.0f} ms" if s.seconds < 1 else f"  {s.seconds:.1f} s",
                    style=MUTED,
                )
            grid.add_row(icon, label, detail)
        return grid


# ------------------------------------------------------------------ retrieval
def score_bar(score: float, width: int = 12, threshold: float = 0.35) -> Text:
    filled = max(0, min(width, round(score * width)))
    colour = TEAL if score >= max(0.5, threshold) else ACCENT if score >= threshold else MUTED
    bar = Text("█" * filled + "░" * (width - filled), style=colour)
    bar.append(f" {score:.2f}", style="bold white")
    return bar


def relevance(hit: Hit) -> float:
    return hit.relevantie if hit.relevantie is not None else hit.score


def retrieval_table(retrieval: Retrieval, hits: Iterable[Hit] | None = None) -> Table:
    table = Table(
        box=box.SIMPLE_HEAD,
        header_style=f"bold {MUTED}",
        show_edge=False,
        pad_edge=False,
        expand=True,
    )
    reranked = retrieval.score_soort == "rerank"
    table.add_column("#", justify="right", width=3, style=MUTED)
    table.add_column("relevantie" if reranked else "cosine", width=18, no_wrap=True)
    if reranked:
        table.add_column("cos", width=5, style=MUTED, no_wrap=True)
    table.add_column("ECLI", style=f"bold {ACCENT}", no_wrap=True)
    table.add_column("instantie", style="white", no_wrap=True)
    table.add_column("datum", style=MUTED, no_wrap=True)
    table.add_column("bron", style=MUTED, ratio=1)
    table.add_column("bm25", justify="center", width=4, style=MUTED)
    for n, hit in enumerate(hits if hits is not None else retrieval.hits, 1):
        paras = (
            f"par. {hit.para_van}"
            if hit.para_van == hit.para_tot
            else f"par. {hit.para_van}-{hit.para_tot}"
        )
        cells = [str(n), score_bar(relevance(hit), threshold=retrieval.drempel)]
        if reranked:
            cells.append(f"{hit.score:.2f}")
        table.add_row(
            *cells,
            hit.ecli,
            hit.instantie.replace("Gerechtshof", "Hof").replace("Rechtbank", "Rb."),
            hit.datum,
            f"§{hit.sectie_nummer} {kort(hit.sectie_titel)}, {paras}",
            "●" if hit.bm25_rank else "",
        )
    return table


def retrieval_panel(retrieval: Retrieval, title: str = "Opgehaalde passages") -> Panel:
    soort = "cross-encoder" if retrieval.score_soort == "rerank" else "cosine"
    sub = (
        f"beste {soort}-score {retrieval.beste_score:.2f} · drempel {retrieval.drempel:.2f} · "
        f"{retrieval.kandidaten} kandidaten (cosine ⊕ BM25)"
        + (f" · {retrieval.herordend} herordend" if retrieval.herordend else "")
        + f" · {retrieval.filters.describe()}"
    )
    return Panel(
        Group(retrieval_table(retrieval), Text(sub, style=MUTED)),
        title=f"[bold white]{title}[/]",
        title_align="left",
        border_style=MUTED,
        box=box.ROUNDED,
        padding=(0, 1),
    )


# ------------------------------------------------------------------ answer
def claim_text(antwoord: Antwoord, upto: int | None = None, partial: str | None = None) -> Text:
    """Claims as rich text; ``upto``/``partial`` drive the typewriter."""
    text = Text()
    claims = antwoord.beweringen if upto is None else antwoord.beweringen[:upto]
    for i, b in enumerate(claims):
        text.append("▎ ", style=TEAL)
        text.append(b.tekst, style="white")
        for v in b.verwijzingen:
            text.append(f" [{v.nummer}]", style=f"bold {ACCENT}")
        if i < len(claims) - 1 or partial is not None:
            text.append("\n")
    if partial is not None:
        text.append("▎ ", style=TEAL)
        text.append(partial, style="white")
        text.append("▌", style=ACCENT)
    return text


def references_table(antwoord: Antwoord) -> Table:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style=f"bold {ACCENT}", no_wrap=True)
    grid.add_column(style="white")
    grid.add_column(style=MUTED)
    seen = set()
    for b in antwoord.beweringen:
        for v in b.verwijzingen:
            if v.nummer in seen:
                continue
            seen.add(v.nummer)
            paras = (
                f"par. {v.para_van}"
                if v.para_van == v.para_tot
                else f"par. {v.para_van}-{v.para_tot}"
            )
            grid.add_row(
                f"[{v.nummer}]",
                Text(v.ecli, style=f"link {v.url} white"),
                f"§{v.sectie_nummer} {kort(v.sectie_titel)}, {paras}",
            )
    return grid


def verification_text(antwoord: Antwoord) -> Text:
    text = Text()
    n_ok, n_bad = len(antwoord.beweringen), len(antwoord.afgekeurd)
    text.append("✔ ", style=f"bold {TEAL}")
    text.append(f"{n_ok} bewering{'en' if n_ok != 1 else ''} geverifieerd", style="white")
    text.append(
        f" · elke ECLI komt uit de {len(antwoord.passages)} aangeleverde passages", style=MUTED
    )
    if antwoord.duplicaten:
        text.append(f" · {antwoord.duplicaten} herhaalde beweringen samengevoegd", style=MUTED)
    if n_bad:
        text.append(f"\n✘ {n_bad} afgekeurd", style=f"bold {RED}")
        for a in antwoord.afgekeurd:
            text.append(f"\n   {a.reden}", style=RED)
            text.append(f"  «{a.tekst[:70]}{'…' if len(a.tekst) > 70 else ''}»", style=MUTED)
    return text


def answer_panel(antwoord: Antwoord, body: Text | None = None) -> Panel:
    parts: list[RenderableType] = [body if body is not None else claim_text(antwoord)]
    if body is None or body.plain.endswith(("]", ".")):
        pass
    parts += [Text(""), references_table(antwoord), Text(""), verification_text(antwoord)]
    parts += [Text(""), Text(antwoord.disclaimer, style=f"italic {MUTED}")]
    return Panel(
        Group(*parts),
        title=f"[bold white]Antwoord[/] [{MUTED}]· {escape(antwoord.llm)}[/]",
        title_align="left",
        border_style=TEAL,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def no_source_panel(result: GeenBron) -> Panel:
    text = Text()
    text.append("Geen bron gevonden\n", style=f"bold {ACCENT}")
    text.append(
        result.tekst.split(". ", 1)[1] if ". " in result.tekst else result.tekst, style="white"
    )
    text.append(f"\n\nreden: {result.reden}\n", style=MUTED)
    gauge = score_bar(result.beste_score, 24, threshold=result.drempel)
    text.append("beste relevantie  ", style=MUTED)
    text.append_text(gauge)
    text.append(f"   drempel {result.drempel:.2f}", style=MUTED)
    if result.kandidaten:
        text.append("\n\ndichtstbijzijnde passages, ter controle:", style=MUTED)
        for h in result.kandidaten[:3]:
            text.append(f"\n  {relevance(h):.2f}  ", style=MUTED)
            text.append(h.ecli, style=ACCENT)
            text.append(f"  §{h.sectie_nummer} {kort(h.sectie_titel, 40)}", style=MUTED)
    text.append("\n\n" + result.disclaimer, style=f"italic {MUTED}")
    return Panel(
        text,
        title="[bold white]Antwoord[/]",
        title_align="left",
        border_style=ACCENT,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def typewriter(antwoord: Antwoord, *, cps: int = 220) -> None:
    """Type the verified claims out, then show references and verification."""
    delay = 1.0 / cps
    with Live(console=console, refresh_per_second=40, transient=True) as live:
        for i, b in enumerate(antwoord.beweringen):
            for j in range(1, len(b.tekst) + 1):
                live.update(_typing_panel(antwoord, claim_text(antwoord, i, b.tekst[:j])))
                time.sleep(delay)
            live.update(_typing_panel(antwoord, claim_text(antwoord, i + 1)))
            time.sleep(0.15)
    console.print(answer_panel(antwoord))


def _typing_panel(antwoord: Antwoord, body: Text) -> Panel:
    return Panel(
        body,
        title=f"[bold white]Antwoord[/] [{MUTED}]· {escape(antwoord.llm)}[/]",
        title_align="left",
        border_style=TEAL,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def error_panel(title: str, detail: str) -> Panel:
    text = Text()
    text.append(f"{title}\n", style=f"bold {RED}")
    text.append(detail, style="white")
    text.append(
        "\n\nEr is geen antwoord gegenereerd. Tip: kies een andere backend met --llm "
        "(ollama | anthropic | extractive) of controleer of Ollama draait.",
        style=MUTED,
    )
    return Panel(
        text, title="[bold white]Fout[/]", title_align="left", border_style=RED, box=box.ROUNDED
    )


def rule(title: str = "") -> Rule:
    return Rule(title, style=MUTED)
