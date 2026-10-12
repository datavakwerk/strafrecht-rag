"""CLI: ingest → index → ask, plus eval, stats and an animated demo."""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from strafrecht_rag import ui
from strafrecht_rag.config import DISCLAIMER, Settings, settings
from strafrecht_rag.index.store import Filters
from strafrecht_rag.query.answer import Antwoord, GeenBron
from strafrecht_rag.ui import ACCENT, MUTED, RED, TEAL, console

app = typer.Typer(
    help="RAG over Dutch criminal-law rulings. Every claim carries an ECLI; no source, no answer.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _filters(
    instantie_type: str | None, instantie: str | None, van: str | None, tot: str | None
) -> Filters:
    return Filters(
        instantie_type=instantie_type,
        instantie=instantie,
        datum_van=date.fromisoformat(van) if van else None,
        datum_tot=date.fromisoformat(tot) if tot else None,
    )


# ---------------------------------------------------------------- ingest
@app.command()
def ingest(
    from_date: Annotated[
        str | None, typer.Option(help="YYYY-MM-DD (default: .env windows)")
    ] = None,
    to_date: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    limit: Annotated[int | None, typer.Option(help="stop after N rulings")] = None,
) -> None:
    """Fetch raw XML from Rechtspraak.nl into data/raw/ (resumable, rate limited)."""
    from strafrecht_rag.ingest.client import RechtspraakClient
    from strafrecht_rag.ingest.client import ingest as run_ingest

    s = settings()
    windows = (
        [(date.fromisoformat(from_date), date.fromisoformat(to_date or from_date))]
        if from_date
        else s.date_windows
    )
    console.print(ui.banner("ingest", ", ".join(f"{a} → {b}" for a, b in windows)))
    counts = {"fetched": 0, "skipped": 0, "error": 0}
    with Progress(
        SpinnerColumn(style=ACCENT),
        TextColumn("[white]{task.description}"),
        BarColumn(bar_width=30, complete_style=TEAL),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("uitspraken ophalen", total=None)
        for event in run_ingest(RechtspraakClient(), s.raw_dir, windows, limit=limit):
            counts[event.status] += 1
            progress.update(
                task,
                advance=1,
                description=f"{event.ecli}  [{MUTED}]nieuw {counts['fetched']} · "
                f"overgeslagen {counts['skipped']} · fouten {counts['error']}[/]",
            )
    console.print(
        f"[{TEAL}]✔[/] {counts['fetched']} opgehaald, {counts['skipped']} al aanwezig, "
        f"{counts['error']} fouten → {s.raw_dir}"
    )


# ---------------------------------------------------------------- index
@app.command()
def index(
    limit: Annotated[int | None, typer.Option(help="index only the first N files")] = None,
    force: Annotated[
        bool, typer.Option("--force", help="re-embed unchanged documents too")
    ] = False,
) -> None:
    """Parse data/raw/, chunk, embed and upsert into the vector store (incremental)."""
    from strafrecht_rag import runtime
    from strafrecht_rag.index.main import index_corpus

    s = settings()
    store = runtime.open_store(s)
    embedder = runtime.embedder_for(s, store)
    config = runtime.chunk_config(s)
    files = sorted(s.raw_dir.glob("*.xml"))
    console.print(
        ui.banner(
            "index",
            f"{len(files)} XML-bestanden · chunks {config.size}/{config.overlap} tekens · "
            f"embedder {embedder.name} ({embedder.dim}d)",
        )
    )
    n_docs = n_chunks = n_err = n_skip = 0
    started = time.perf_counter()
    with Progress(
        SpinnerColumn(style=ACCENT),
        TextColumn("[white]{task.description}"),
        BarColumn(bar_width=30, complete_style=TEAL),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("indexeren", total=min(len(files), limit or len(files)))
        for event in index_corpus(
            s.raw_dir, s.parsed_dir, store, embedder, config, limit=limit, force=force
        ):
            if event.status == "indexed":
                n_docs += 1
                n_chunks += event.chunks
            elif event.status == "skipped":
                n_skip += 1
            else:
                n_err += 1
            rate = n_chunks / max(time.perf_counter() - started, 1e-6)
            progress.update(
                task,
                advance=1,
                description=f"{event.ecli}  [{MUTED}]{n_chunks} chunks · {rate:.0f}/s · "
                f"{n_skip} ongewijzigd · {n_err} parsefouten[/]",
            )
    stats = store.stats()
    table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    table.add_column(style=MUTED)
    table.add_column(style="bold white")
    table.add_row(
        "uitspraken geïndexeerd", f"{n_docs}  ({n_skip} ongewijzigd, {n_err} parsefouten)"
    )
    table.add_row("chunks in store", f"{stats['chunks']}")
    table.add_row("per type", ", ".join(f"{k} {v}" for k, v in stats["per_type"].items()))
    table.add_row("per jaar", ", ".join(f"{k} {v}" for k, v in stats["per_jaar"].items()))
    table.add_row("store", str(store.path))
    console.print(table)


# ---------------------------------------------------------------- ask
@app.command()
def ask(
    vraag: Annotated[str, typer.Argument(help="the question, in Dutch")],
    k: Annotated[int | None, typer.Option("--k", help="passages to retrieve")] = None,
    instantie_type: Annotated[
        str | None, typer.Option("--type", help="Rechtbank | Gerechtshof | Hoge Raad")
    ] = None,
    instantie: Annotated[str | None, typer.Option(help="court name contains …")] = None,
    van: Annotated[str | None, typer.Option(help="from date YYYY-MM-DD")] = None,
    tot: Annotated[str | None, typer.Option(help="to date YYYY-MM-DD")] = None,
    llm: Annotated[str | None, typer.Option(help="ollama | anthropic | extractive")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="machine-readable output")] = False,
    animate: Annotated[bool, typer.Option(help="typewriter + spinners")] = True,
) -> None:
    """Answer a question from the corpus, every claim with an ECLI reference."""
    s = settings()
    filters = _filters(instantie_type, instantie, van, tot)
    if as_json:
        result = _answer_plain(s, vraag, k, filters, llm)
        console.print_json(json.dumps(result.to_dict(), ensure_ascii=False))
        return
    _answer_animated(s, vraag, k, filters, llm, animate=animate and console.is_terminal)


def _answer_plain(s: Settings, vraag: str, k: int | None, filters: Filters, llm: str | None):
    from strafrecht_rag import runtime

    generator, _ = runtime.generator_for(s, llm=llm)
    return generator.answer(vraag, k=k, filters=filters)


def _answer_animated(
    s: Settings,
    vraag: str,
    k: int | None,
    filters: Filters,
    llm: str | None,
    *,
    animate: bool,
    prebuilt: tuple | None = None,
) -> Antwoord | GeenBron:
    from strafrecht_rag import runtime

    console.print(ui.banner("ask", vraag))
    stages = ui.Stages()
    labels = [
        "Componenten laden",
        "Vraag embedden",
        "Passages ophalen",
        "Herordenen",
        "Antwoord genereren",
        "Bronverificatie",
    ]
    for label in labels:
        stages.add(label)
    with Live(stages, console=console, refresh_per_second=20, transient=False) as live:
        stages.start(0)
        generator, store = prebuilt or runtime.generator_for(s, llm=llm)
        retriever = generator.retriever
        emb = retriever.embedder
        rr = retriever.reranker
        stages.done(
            0,
            f"{store.count()} chunks · {emb.name.split('/')[-1]}"
            + (f" · {rr.name.split('/')[-1]}" if rr else "")
            + f" · {generator.llm.name}",
        )

        def on_stage(name: str) -> None:
            if name == "embed":
                stages.start(1)
            elif name == "search":
                stages.done(1, f"{emb.dim}-dimensionale vector")
                stages.start(2)
            elif name == "rerank":
                stages.done(2, "cosine ⊕ BM25, reciprocal rank fusion")
                stages.start(3)
            live.refresh()

        retrieval = generator.retrieve(vraag, k=k, filters=filters, on_stage=on_stage)
        if retrieval.score_soort == "rerank":
            stages.done(3, f"{retrieval.herordend} kandidaten door de cross-encoder")
        else:
            stages.done(2, "cosine ⊕ BM25, reciprocal rank fusion")
            stages.skip(3, "geen reranker geconfigureerd")
        detail = f"top-{len(retrieval.hits)} · {filters.describe()}"
        stages.stages[2].detail = f"{stages.stages[2].detail} · {detail}"
        llm_error: Exception | None = None
        if not retrieval.boven_drempel:
            stages.skip(
                4,
                f"overgeslagen: beste relevantie {retrieval.beste_score:.2f} "
                f"< drempel {retrieval.drempel:.2f}",
            )
            stages.skip(5, "niets te verifiëren")
            live.refresh()
            result: Antwoord | GeenBron = generator.answer_from(retrieval)
        else:
            stages.start(4)
            live.refresh()
            try:
                raw = generator.generate(vraag, retrieval)
            except Exception as exc:  # network, timeout, missing model: report, don't crash
                stages.fail(4, f"{type(exc).__name__}: {str(exc)[:80] or 'geen details'}")
                stages.skip(5, "niets te verifiëren")
                live.refresh()
                llm_error = exc
                raw = None
            else:
                llm_error = None
                stages.done(4, f"{len(raw)} tekens ruwe output")
        if llm_error is not None:
            console.print()
            if retrieval.hits:
                console.print(ui.retrieval_panel(retrieval))
            console.print(ui.error_panel(f"{generator.llm.name} gaf geen antwoord", str(llm_error)))
            raise typer.Exit(1)
        if retrieval.boven_drempel:
            stages.start(5)
            live.refresh()
            result = generator.answer_from(retrieval, raw)
            if isinstance(result, Antwoord):
                stages.done(
                    5, f"{len(result.beweringen)} geverifieerd · {len(result.afgekeurd)} afgekeurd"
                )
            else:
                stages.done(5, "model meldt GEEN_BRON")
    console.print()
    if retrieval.hits:
        console.print(ui.retrieval_panel(retrieval))
    elif isinstance(result, GeenBron):
        pass
    console.print()
    if isinstance(result, GeenBron):
        console.print(ui.no_source_panel(result))
    elif animate:
        ui.typewriter(result)
    else:
        console.print(ui.answer_panel(result))
    return result


# ---------------------------------------------------------------- eval
@app.command("eval")
def eval_cmd(
    k: Annotated[int | None, typer.Option("--k")] = None,
    llm: Annotated[str | None, typer.Option(help="ollama | anthropic | extractive")] = None,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="retrieval metrics only")] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    path: Annotated[Path | None, typer.Option(help="question set")] = None,
) -> None:
    """Recall@k, MRR, source correctness and no-source correctness on the question set."""
    from strafrecht_rag import runtime
    from strafrecht_rag.eval import EVAL_PATH, evaluate, load_questions

    s = settings()
    k = k or s.top_k
    vragen = load_questions(path or EVAL_PATH)
    generator, store = runtime.generator_for(s, llm=llm)
    if not as_json:
        console.print(
            ui.banner(
                "eval",
                f"{len(vragen)} vragen · k={k} · drempel {s.score_threshold} · "
                f"{store.count()} chunks · llm {'—' if no_llm else generator.llm.name}",
            )
        )
    table = Table(box=box.SIMPLE_HEAD, header_style=f"bold {MUTED}", show_edge=False, expand=True)
    table.add_column("id", style=MUTED, width=5)
    table.add_column("type", style=MUTED, width=11)
    table.add_column("vraag", style="white", ratio=1)
    table.add_column("score", width=18)
    table.add_column("recall", justify="right", width=6)
    table.add_column("rr", justify="right", width=5)
    table.add_column("claims", justify="right", width=7)
    table.add_column("status", width=9)

    def on_result(r):
        status_style = TEAL if r.status == "ok" else RED if r.status in {"fout", "mis"} else ACCENT
        table.add_row(
            r.vraag.id,
            r.vraag.type,
            r.vraag.vraag[:70] + ("…" if len(r.vraag.vraag) > 70 else ""),
            ui.score_bar(r.beste_score),
            "—" if r.recall is None else f"{r.recall:.2f}",
            "—" if r.reciprocal_rank is None else f"{r.reciprocal_rank:.2f}",
            "—" if not r.claims_raw else f"{r.claims_ok}/{r.claims_raw}",
            Text(r.status, style=f"bold {status_style}"),
        )

    if as_json:
        report = evaluate(generator, vragen, k=k, with_llm=not no_llm)
        console.print_json(json.dumps(report.to_dict(), ensure_ascii=False))
        return
    with Live(table, console=console, refresh_per_second=8):
        report = evaluate(generator, vragen, k=k, with_llm=not no_llm, on_result=on_result)
    console.print()
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style=MUTED, min_width=28)
    summary.add_column(min_width=30)
    summary.add_column(style=MUTED)

    def metric(label: str, value: float, note: str = ""):
        summary.add_row(label, ui.score_bar(value, 20), note)

    n_a, n_n = len(report.answerable), len(report.no_source)
    metric(f"recall@{k}", report.recall_at_k, f"{n_a} beantwoordbare vragen")
    metric(f"hit@{k}", report.hit_at_k, "minstens één verwachte ECLI in top-k")
    metric("MRR", report.mrr, "rang van de eerste verwachte ECLI")
    if not no_llm:
        metric(
            "bronjuistheid", report.bronjuistheid, "elke ECLI in het antwoord komt uit de passages"
        )
        metric(
            "claim-acceptatie",
            report.claim_acceptance,
            "ruwe LLM-beweringen die verificatie doorstaan",
        )
    metric(
        "geen-bron-correctheid",
        report.geen_bron_correctheid,
        f"{n_n} vragen zonder bron in het corpus",
    )
    metric(
        "onterecht geen-bron",
        report.onterecht_geen_bron,
        "beantwoordbare vragen zonder antwoord (lager is beter)",
    )
    console.print(
        Panel(
            summary,
            title="[bold white]Resultaat[/]",
            title_align="left",
            border_style=TEAL,
            box=box.ROUNDED,
        )
    )


# ---------------------------------------------------------------- stats
@app.command()
def stats() -> None:
    """Corpus and index statistics."""
    from strafrecht_rag import runtime

    s = settings()
    store = runtime.open_store(s)
    st = store.stats()
    raw = len(list(s.raw_dir.glob("*.xml"))) if s.raw_dir.exists() else 0
    parsed = len(list(s.parsed_dir.glob("*.json"))) if s.parsed_dir.exists() else 0
    console.print(ui.banner("stats", str(s.data_dir.resolve())))
    table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    table.add_column(style=MUTED)
    table.add_column(style="bold white")
    table.add_row("ruwe XML's", str(raw))
    table.add_row("geparst", str(parsed))
    table.add_row("uitspraken in index", str(st["uitspraken"]))
    table.add_row("chunks", str(st["chunks"]))
    table.add_row("instanties", str(st["instanties"]))
    table.add_row("per type", ", ".join(f"{k} {v}" for k, v in st["per_type"].items()) or "—")
    table.add_row("per jaar", ", ".join(f"{k} {v}" for k, v in st["per_jaar"].items()) or "—")
    table.add_row("embedder", store.get_meta("embedder") or "—")
    table.add_row(
        "chunking",
        f"{store.get_meta('chunk_size') or '—'} / {store.get_meta('chunk_overlap') or '—'} tekens",
    )
    console.print(table)


# ---------------------------------------------------------------- demo
DEMO_QUESTIONS: list[tuple[str, str, dict]] = [
    (
        "semantisch",
        "Welke uitspraken gaan over verduistering door een bestuurder van een stichting?",
        {},
    ),
    (
        "gefilterd",
        "Welke uitspraken gaan over handel in vuurwapens en munitie?",
        {"instantie_type": "Rechtbank", "van": "2024-01-01"},
    ),
    (
        "inhoudelijk",
        "Wat woog de rechter mee bij de strafmaat voor de woningoverval in Alteveer?",
        {},
    ),
    ("eerlijk", "Wat zegt de rechter over de octrooiaanvraag voor een kernfusiereactor?", {}),
]


@app.command()
def demo(
    pause: Annotated[float, typer.Option(help="seconds between questions")] = 2.5,
    llm: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Walk through the four question types from the PRD, animated."""
    from strafrecht_rag import runtime

    s = settings()
    prebuilt = runtime.generator_for(s, llm=llm)
    for i, (kind, vraag, f) in enumerate(DEMO_QUESTIONS, 1):
        console.clear()
        console.print(
            Panel(
                Text.assemble(
                    (f"{i}/{len(DEMO_QUESTIONS)}  ", MUTED),
                    (kind, f"bold {ACCENT}"),
                    (
                        "  ·  "
                        + {
                            "semantisch": "betekenis, niet trefwoorden",
                            "gefilterd": "alleen gerechtshoven, 2024–2025",
                            "inhoudelijk": "wat zegt de rechter precies?",
                            "eerlijk": "geen bron → geen antwoord",
                        }[kind],
                        MUTED,
                    ),
                ),
                box=box.HEAVY,
                border_style=ACCENT,
                padding=(0, 2),
            )
        )
        time.sleep(1.2)
        filters = _filters(f.get("instantie_type"), None, f.get("van"), f.get("tot"))
        _answer_animated(s, vraag, None, filters, llm, animate=True, prebuilt=prebuilt)
        time.sleep(pause)
    console.print(f"\n[{MUTED}]{DISCLAIMER}[/]")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
