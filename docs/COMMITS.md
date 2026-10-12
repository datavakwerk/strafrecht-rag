# Commitplan — De eerste 10 commits — strafrecht-rag

Companion bij [PRD.md](PRD.md). Zelfde regels als de andere twee plannen: één
sitting per commit, repo blijft werkend, groen vóór je commit. Dit document staat
bewust **buiten** de repo — werkplan, geen deliverable.

**Scope van deze 10 commits:** de volledige keten van ophalen tot beantwoorden,
inclusief evaluatieset en CI. Wat erna komt: corpus opschalen, retrieval-tuning op
de evaluatiecijfers, decisions-log en README-polish.

**Twee beslissingen die dit plan vooruitschuift** (bewust, zie PRD §10): het
embedding-model en de vector store worden pas in commit 6 gekozen, achter de
interface uit commit 5/6. Alles t/m commit 5 is er niet van afhankelijk.

**Conventies:** Conventional Commits; "Klaar wanneer" = handmatige controle vóór
het committen; paden relatief aan de repowortel.

---

## Commit 1 — Project-scaffolding

> `chore: project scaffolding`

**Doel.** De PRD §5.3-indeling, leeg maar kloppend. Repo initialiseren met
`README.md` (bestaat al — bijwerken naar de definitieve indeling), `.gitignore`
(`data/`, `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`), licentie, en de
mappen `src/{ingest,index,query}` en `tests/`.

**Klaar wanneer.** `git status` schoon na een verse clone; mappenstructuur komt
overeen met PRD §5.3.

---

## Commit 2 — Dependencies en tooling

> `chore: dependency management en tooling`

**Doel.** Package-manager en lockfile vastleggen, ruff + pytest, en een Makefile
met `install`, `lint`, `test`. Zelfde keuzes als de andere portfoliorepo's:
Python 3.12, ruff-config in `pyproject.toml`.

**Bestanden.** `pyproject.toml`, lockfile, `Makefile`,
`tests/test_smoke.py` (import-smoke per package).

**Klaar wanneer.** `make install && make lint && make test` groen op een verse
clone.

---

## Commit 3 — Ingestie: uitspraken ophalen van Rechtspraak.nl

> `feat(ingest): uitspraken ophalen van Rechtspraak.nl`

**Doel.** Client die het Open Data-endpoint bevraagt op datum, instantie en
rechtsgebied, en ruwe XML per ECLI wegschrijft naar `data/raw/<ECLI>.xml`. Rate
limiting en retries vanaf het begin — een nette client is geen polish maar een
voorwaarde.

**Bestanden.**

```
src/ingest/client.py        # zoekindex bevragen + document per ECLI ophalen
src/ingest/main.py          # CLI-aanroepbare run: periode/instantie/rechtsgebied
tests/fixtures/zoek_*.xml   # vastgelegde zoekresponses
tests/test_client.py        # paginering, retry-gedrag (gemockte responses)
```

**Details.**

- Al opgehaalde ECLI's overslaan (bestand bestaat) → herstart hervat zonder
  dubbelwerk (PRD DoD).
- **Beslis hier PRD OQ3** (corpus-afbakening): meet hoeveel uitspraken de gekozen
  subset oplevert en leg het aantal vast voor de README.
- Fixtures direct vastleggen: een script/aantekening in `tests/fixtures/` hoe ze
  opgehaald zijn.

**Klaar wanneer.** Een run op een kleine periode vult `data/raw/` met geldige
XML's; herstart haalt niets dubbel op; tests groen zonder netwerk.

---

## Commit 4 — Parser: XML → gestructureerd document

> `feat(ingest): parser van XML naar gestructureerd document`

**Doel.** ECLI, instantie, datum, rechtsgebied, procedure en tekstsecties uit de
XML naar één genormaliseerd `Uitspraak`-model. Dit is het datacontract waar index
en query op bouwen — het meest ontwerpzware commit van de tien.

**Bestanden.**

```
src/ingest/model.py           # dataclass Uitspraak + Sectie
src/ingest/parser.py          # XML → Uitspraak; per-document foutafhandeling
tests/fixtures/ecli_*.xml     # diverse echte uitspraken (instanties, jaargangen)
tests/test_parser.py
```

**Details.**

- **Verzamel bewust diverse fixtures** (rechtbank én hof, oud én nieuw) — de
  XML-structuur varieert en dat is het hoogste risico uit de PRD-risicotabel.
- Eén onparsbaar document levert één foutrecord, nooit een gecrashte batch.
- Sectie- en paragraafnummers behouden: commit 5 heeft ze nodig voor
  herleidbaarheid.

**Klaar wanneer.** `make test` groen; een scratch-run parseert alles in
`data/raw/` en rapporteert het foutpercentage (noteer het voor de README).

---

## Commit 5 — Chunking van uitspraaktekst

> `feat(index): chunking van uitspraaktekst`

**Doel.** Tekst opdelen in overlappende chunks met behoud van sectie- en
paragraafnummers, zodat een chunk exact terug te voeren is op de bron. Puur en
deterministisch — geen embeddings, geen store, geen netwerk. Dé kwaliteitsknop
van het hele systeem, dus getest als een bibliotheek.

**Bestanden.**

```
src/index/chunker.py       # Chunk-dataclass + chunk(uitspraak, config)
tests/test_chunker.py      # overlap, sectiegrenzen, korte secties, lege secties
```

**Details.**

- Chunkgrootte en overlap als config, niet als constante — commit 10 maakt het
  effect meetbaar.
- Een chunk draagt ECLI, sectie, paragraafbereik en positie (PRD-datacontract);
  een chunk zonder exacte bronpositie is een bug.
- Randgevallen expliciet testen: sectie korter dan de chunkgrootte, sectiegrens
  midden in een overlap.

**Klaar wanneer.** `make test` groen; chunks van een echte uitspraak handmatig
teruggezocht in de brontekst.

---

## Commit 6 — Embeddings en vector store

> `feat(index): embeddings en vector store`

**Doel.** Embedding-model achter een interface, chunks batchgewijs embedden en
opslaan in de vector store, inclusief metadata (ECLI, instantie, datum, sectie).
**Hier vallen de beslissingen van PRD OQ1 en OQ2** — timebox ze; de interface
maakt terugkomen goedkoop.

**Bestanden.**

```
src/index/embeddings.py     # interface + gekozen backend + FakeEmbedder (tests)
src/index/store.py          # vector store: upsert, query met filters
src/index/main.py           # indexeer-run over data/
tests/test_store.py         # upsert-idempotentie, metadatafilters (met fake)
```

**Details.**

- Tests draaien volledig op de `FakeEmbedder` (deterministische vectoren) — geen
  model en geen netwerk in CI (PRD N1).
- Herindexeren idempotent: upsert op chunk-id (ECLI + positie), geen dubbelen.
- Leg de modelkeuze en de store-keuze direct vast in de decisions-log.

**Klaar wanneer.** `make test` groen; een echte indexeer-run over het corpus
draait door en het aantal chunks klopt met de chunker-output.

---

## Commit 7 — Retrieval

> `feat(query): retrieval`

**Doel.** Vraag embedden, top-k chunks ophalen met filters op instantie, datum en
rechtsgebied, en resultaten teruggeven met score en bronmetadata.

**Bestanden.**

```
src/query/retriever.py      # top-k + filters + scoredrempel
tests/test_retriever.py     # met fake embedder + kleine geseede store
```

**Details.**

- De scoredrempel is hier al aanwezig: onder de drempel → leeg resultaat. Dat is
  de bouwsteen waarop het geen-bron-pad van commit 8 rust.
- Resultaten dragen alles wat generatie nodig heeft: tekst, score, ECLI, sectie,
  paragraafbereik.

**Klaar wanneer.** `make test` groen; een handvraag op het echte corpus geeft
herkenbaar relevante passages met plausibele scores.

---

## Commit 8 — Antwoordgeneratie met bronvermelding

> `feat(query): antwoordgeneratie met bronvermelding`

**Doel.** Het thesis-commit van dit project (zoals commit 9 dat is bij de
OV-pipeline): antwoorden uitsluitend op basis van aangeleverde passages, ECLI per
bewering, en een expliciet geen-bron-pad — afgedwongen in code, niet alleen in de
prompt.

**Bestanden.**

```
src/query/generator.py      # prompt-opbouw, LLM-interface + fake, bronverificatie
src/query/answer.py         # Antwoord-model: beweringen + verwijzingen | GeenBron
tests/test_generator.py     # bronverificatie, geen-bron-pad (met fake LLM)
```

**Details.**

- **Bronverificatie in code (PRD N2):** elke ECLI in het antwoord moet voorkomen
  in de aangeleverde passages; een bewering zonder geldige verwijzing wordt
  afgekeurd. Test met een fake LLM die expres een verzonnen ECLI teruggeeft.
- Lege retrieval (drempel van commit 7) slaat de LLM volledig over → het
  geen-bron-antwoord is een codepad, geen promptgedrag.
- Disclaimer "geen juridisch advies" in de output (PRD N4).

**Klaar wanneer.** `make test` groen, inclusief de verzonnen-ECLI-test en het
geen-bron-pad.

---

## Commit 9 — CLI: end-to-end commando's

> `feat(cli): end-to-end commando's`

**Doel.** `ingest`, `index` en `ask` als CLI-commando's, zodat de hele keten van
ophalen tot antwoorden vanaf de commandline draait. Sluit de tien commits af zoals
de andere projecten: de kernbelofte end-to-end demonstreerbaar.

**Bestanden.**

```
src/cli.py                  # argumentparsing → de drie ketens
Makefile                    # ingest / index / ask / eval targets
tests/test_cli.py           # argumentrouting (kern gemockt)
```

**Details.**

- `ask` toont per bewering de ECLI-verwijzing en de gebruikte passages; `--json`
  voor machineleesbare output (het evaluatiescript van commit 10 gebruikt die).
- Config (corpus-afbakening, chunkparameters, drempel, backends) via `.env`/
  flags — geen hardcoded keuzes.

**Klaar wanneer.** Op een verse clone: `make install`, dan `ingest` → `index` →
`ask` met een echte vraag, antwoord met bronnen op het scherm.

---

## Commit 10 — Evaluatieset en basis-CI

> `test: evaluatieset en basis-CI`

**Doel.** De cijfers waarmee je het project verdedigt, plus de CI die alles
hierna bewaakt. Kleine set vragen met verwachte ECLI's, een script dat recall@k
en bronjuistheid meet, en een workflow die lint en tests draait op elke push.

**Bestanden.**

```
tests/fixtures/eval_vragen.json   # vragen → verwachte ECLI's; incl. geen-bron-vragen
scripts/eval.py                   # recall@k, bronjuistheid, geen-bron-correctheid
.github/workflows/ci.yml          # make lint && make test; badge in README
```

**Details.**

- **De evaluatieset bevat ook vragen zonder bron in het corpus** — het geen-
  bron-pad wordt gemeten, niet aangenomen.
- CI draait zonder netwerk en zonder LLM (fakes + fixtures, PRD N1); de echte
  evaluatie (`make eval`) is een lokaal commando en de cijfers gaan in de README.
- Zelfde les als bij open-data-warehouse: bewijs op een wegwerpbranch dat CI ook
  echt rood kán worden.

**Klaar wanneer.** CI groen op GitHub met badge; `make eval` print recall@k en
bronjuistheid; de cijfers + vragenset-omvang staan in de README.

---

## Na commit 10 (vooruitblik, niet in detail gepland)

Corpus opschalen naar de volledige afbakening → chunkparameters en drempel tunen
op de evaluatiecijfers (nu meetbaar) → decisions-log en README-polish (cijfers,
voorbeeldsessie, disclaimer) → optioneel: reranking of hybride zoeken (BM25 +
vectoren) als de recall@k erom vraagt — pas ná gemeten noodzaak.

## Beslissingen die dit plan al neemt (veto ze vroeg als je het er niet mee eens bent)

| # | Beslissing | Waar |
|---|---|---|
| 1 | Python 3.12, ruff + pytest, Makefile — zelfde tooling als de andere repo's | Commit 2 |
| 2 | Corpus-afbakening op basis van gemeten volume (PRD OQ3) | Commit 3 |
| 3 | Chunking als eigen, pure module — geen framework-black-box | Commit 5 |
| 4 | Embedding-model en vector store pas in commit 6, achter interfaces (OQ1/OQ2) | Commit 6 |
| 5 | Scoredrempel in retrieval; geen-bron is een codepad | Commits 7–8 |
| 6 | Bronverificatie in code, niet alleen in de prompt (PRD N2) | Commit 8 |
| 7 | CI zonder netwerk/LLM; echte evaluatie lokaal via `make eval` | Commit 10 |
