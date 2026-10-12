# PRD — Strafrecht-RAG (Rechtspraak.nl)

**Status:** Concept
**Eigenaar:** Ruud Juffermans
**Aangemaakt:** 2026-08-12
**Repo:** `strafrecht-rag` (alleen README; nog geen code)
**Werkplan:** [COMMITS.md](COMMITS.md) · **Voortgang:** [checklist.md](checklist.md)

---

## 1. Overzicht

### 1.1 Eén zin

RAG-systeem (Retrieval-Augmented Generation) over Nederlandse rechterlijke
uitspraken: uitspraken ophalen via Rechtspraak.nl Open Data, chunken en indexeren
in een vector store, en vragen beantwoorden uitsluitend op basis van opgehaalde
passages — altijd met ECLI-verwijzing, en zonder bron géén antwoord.

### 1.2 Waarom dit project

- RAG is dé standaardvraag in LLM-engineeringrollen; een werkend systeem met
  meetbare retrieval-kwaliteit onderscheidt zich van de zoveelste chatbot-demo.
- Juridische bronnen dwingen de moeilijke eisen af die het project interessant
  maken: verifieerbare bronvermelding, een expliciet "geen bron gevonden"-pad, en
  chunks die exact terug te voeren zijn op sectie en paragraaf.
- Rechtspraak.nl Open Data is vrij toegankelijk, Nederlandstalig en omvangrijk —
  een reëel corpus, geen speelgoedset.

### 1.3 Gouden regel

**Geen antwoord verzinnen.** Elke bewering in een antwoord draagt een
ECLI-verwijzing naar een daadwerkelijk opgehaalde passage; zonder relevante bron
antwoordt het systeem expliciet dat het geen bron heeft. Die eigenschap wordt
niet beloofd maar gemeten (bronjuistheid in de evaluatieset, commit 10).

---

## 2. Doelen & Niet-doelen

### 2.1 Doelen

1. Ingestie-client voor Rechtspraak.nl Open Data: bevragen op datum, instantie en
   rechtsgebied (strafrecht), ruwe XML per ECLI naar `data/raw/`, met rate
   limiting en retries.
2. Parser van XML naar één genormaliseerd `Uitspraak`-model: ECLI, instantie,
   datum, rechtsgebied, procedure en tekstsecties.
3. Chunking met overlap en behoud van sectie-/paragraafnummers, zodat elke chunk
   exact herleidbaar is.
4. Embeddings achter een interface; chunks batchgewijs geïndexeerd in een vector
   store met metadata (ECLI, instantie, datum, sectie).
5. Retrieval: vraag embedden, top-k met metadatafilters, resultaten met score en
   bron.
6. Antwoordgeneratie die aantoonbaar alleen op aangeleverde passages steunt, met
   ECLI per bewering en een expliciet geen-bron-pad.
7. CLI die de hele keten draait: `ingest`, `index`, `ask`.
8. Evaluatieset (vragen → verwachte ECLI's) met een script dat recall@k en
   bronjuistheid meet; lint + tests in CI op elke push.

### 2.2 Niet-doelen

- Geen juridisch advies — de output is een zoek-/samenvattingshulpmiddel met
  bronnen, en zegt dat ook.
- Geen webinterface — de CLI is het product; een UI kan later.
- Geen fine-tuning van modellen — retrieval-kwaliteit is de knop waaraan gedraaid
  wordt, niet het model.
- Geen volledige Rechtspraak-corpus — een afgebakende subset (strafrecht, recente
  jaren, geselecteerde instanties) is genoeg om alles te bewijzen.
- Geen eigen vector-database-infrastructuur — een embedded/lokale store volstaat;
  de interface maakt wisselen mogelijk.

---

## 3. Gebruikers & Vragen

| Persona | Wat die wil |
|---|---|
| **Opdrachtgever / recruiter** | Bewijs van RAG-engineering: chunkingstrategie, retrieval-metrics, hallucinatiebeheersing — met cijfers. |
| **Jurist / onderzoeker (gesimuleerd)** | "Welke uitspraken gaan over X, en wat zegt de rechter daar precies?" — met klikbare ECLI's. |

Vragen die het systeem moet kunnen beantwoorden:

1. Semantisch: "uitspraken over verduistering door een penningmeester" — ook als
   die woorden er niet letterlijk in staan.
2. Gefilterd: idem, maar alleen gerechtshoven, alleen 2023–2025.
3. Inhoudelijk: "wat woog de rechter mee bij de strafmaat in ECLI:…?" — antwoord
   opgebouwd uit passages, per bewering een verwijzing.
4. Eerlijk: een vraag waarvoor het corpus geen bron heeft → expliciet "geen bron
   gevonden", geen gefabuleerd antwoord.

---

## 4. Data

| Bron | Data | Volume | Auth | Notities |
|---|---|---|---|---|
| **Rechtspraak.nl Open Data** | uitspraken en metadata per ECLI (XML); zoekindex op datum/instantie/rechtsgebied | honderdduizenden uitspraken; subset: strafrecht, af te bakenen periode | geen sleutel | uitspraken zijn geanonimiseerd gepubliceerd; ruwe XML blijft buiten git |

**Datacontract:** elke ruwe XML wordt onveranderd bewaard in `data/raw/<ECLI>.xml`
vóór parsing. Het geparste `Uitspraak`-model draagt: ECLI, instantie, datum,
rechtsgebied, procedure, secties met paragraafnummers. Elke chunk draagt: ECLI,
sectie, paragraafbereik, positie — genoeg om de bronpassage exact terug te vinden.

---

## 5. Architectuur

```
  Rechtspraak.nl Open Data
        │  ingest: query op datum/instantie/rechtsgebied,
        │          ruwe XML per ECLI, rate limiting + retries
        ▼
   data/raw/<ECLI>.xml
        │  parse: XML → Uitspraak (ECLI, secties, metadata)
        ▼
   data/parsed/
        │  index: chunking (overlap, sectiebehoud)
        │         → embeddings (achter interface)
        ▼
   vector store (chunks + metadata)
        │  query: vraag embedden → top-k + filters
        ▼
   generatie: antwoord uitsluitend uit passages,
              ECLI per bewering, geen-bron-pad
        ▼
   CLI: ingest / index / ask        eval: recall@k, bronjuistheid
```

### 5.1 Componenten

| Component | Technologie | Rationale |
|---|---|---|
| Taal & tooling | Python 3.12, ruff + pytest, Makefile | zelfde keuzes als de andere portfoliorepo's |
| Ingestie | `httpx`-client op het Open Data-endpoint | dun, testbaar; fixtures van echte XML's in de repo |
| Parsing | `lxml`/`ElementTree` → dataclass `Uitspraak` | XML-eigenaardigheden geïsoleerd in één module |
| Chunking | eigen module: overlappende chunks, sectie-/paragraafbehoud | dé kwaliteitsknop van RAG; verdient eigen tests, geen framework-black-box |
| Embeddings | interface met verwisselbare backend (lokaal model óf API) | kosten en reproduceerbaarheid; keuze is OQ1 |
| Vector store | embedded store met metadatafilters (keuze is OQ2) | geen serverinfrastructuur; interface maakt wisselen goedkoop |
| Generatie | LLM achter interface; prompt dwingt bronplicht af | het geen-bron-pad is een codepad, geen prompthoop |
| CLI | `ingest` / `index` / `ask` | de hele keten demonstreerbaar vanaf de commandline |
| Evaluatie | vaste vragenset + script: recall@k, bronjuistheid | de cijfers waarmee je het project verdedigt |

### 5.2 Ontwerpprincipes

- **Fixture-first:** parsing, chunking en retrieval-logica getest op vastgelegde
  voorbeeld-XML's; geen netwerk en geen LLM nodig in CI.
- **Interfaces op de duurste afhankelijkheden** (embeddings, LLM): tests draaien
  met een fake; de echte backend is configuratie.
- **Herleidbaarheid boven alles:** een chunk zonder exacte bronpositie is een bug.
- **Het geen-bron-pad is een feature:** onder een score-drempel wordt niet
  gegenereerd maar eerlijk "geen bron gevonden" teruggegeven — en de evaluatieset
  bevat vragen die dit pad hóren te raken.

### 5.3 Repository-indeling

```
strafrecht-rag/
├── README.md
├── Makefile                 # install / lint / test / ingest / index / ask / eval
├── pyproject.toml
├── data/                    # ruwe en verwerkte uitspraken (niet in git)
├── src/
│   ├── ingest/              # client + parser
│   ├── index/               # chunking, embeddings, vector store
│   └── query/               # retrieval + generatie
├── tests/
│   └── fixtures/            # voorbeeld-XML's, evaluatievragen
└── .github/workflows/ci.yml
```

---

## 6. Requirements

### 6.1 Functioneel

| ID | Requirement |
|---|---|
| F1 | Ingestie op datum/instantie/rechtsgebied; ruwe XML per ECLI naar `data/raw/`; rate limiting + retries |
| F2 | Parser: XML → `Uitspraak` met ECLI, instantie, datum, rechtsgebied, procedure, tekstsecties |
| F3 | Chunking met overlap; elke chunk herleidbaar tot ECLI + sectie + paragraaf |
| F4 | Embeddings batchgewijs; vector store met metadata; herindexeren idempotent |
| F5 | Retrieval: top-k met filters op instantie/datum/rechtsgebied; score + bron per resultaat |
| F6 | Generatie uitsluitend uit aangeleverde passages; ECLI per bewering; expliciet geen-bron-pad |
| F7 | CLI: `ingest`, `index`, `ask` draaien de keten end-to-end |
| F8 | Evaluatiescript meet recall@k en bronjuistheid op een vaste vragenset |

### 6.2 Niet-functioneel

| ID | Requirement |
|---|---|
| N1 | CI zonder netwerk, zonder LLM-calls: fixtures + fakes (zelfde principe als de andere projecten) |
| N2 | Geverifieerde bronvermelding: elke ECLI in een antwoord komt uit de daadwerkelijk opgehaalde passages (gecontroleerd in code, niet alleen in de prompt) |
| N3 | Persoonsgegevens: alleen de geanonimiseerde publicatie gebruiken; niets de-anonimiseren of verrijken |
| N4 | Disclaimer "geen juridisch advies" in CLI-output en README |
| N5 | Elke niet-triviale keuze in een decisions-log in de README |

---

## 7. Definition of Done

- [ ] `make install && make lint && make test` groen op een verse clone
- [ ] `ingest` haalt een afgebakende strafrecht-subset op; herstart hervat zonder dubbelwerk
- [ ] `index` bouwt de vector store reproduceerbaar op uit `data/`
- [ ] `ask` beantwoordt de vier vraagtypen uit §3, aantoonbaar met bronnen
- [ ] De geen-bron-vraag uit de evaluatieset geeft het geen-bron-antwoord
- [ ] Recall@k en bronjuistheid gemeten en in de README, met de vragenset in de repo
- [ ] CI (lint + tests) groen op elke push; repo publiek met badge

---

## 8. Mijlpalen

| Fase | Deliverable | Commits |
|---|---|---|
| **Fundament** | scaffold, tooling, CI-basis | 1–2 |
| **Ingestie** | client + parser, fixture-getest | 3–4 |
| **Index** | chunking, embeddings, vector store | 5–6 |
| **Query** | retrieval, generatie met bronplicht | 7–8 |
| **Keten & bewijs** | CLI end-to-end, evaluatieset + CI | 9–10 |

---

## 9. Risico's & Mitigaties

| Risico | Kans | Mitigatie |
|---|---|---|
| XML-structuur varieert per instantie/jaargang | Hoog | fixture-first: verzamel vroeg diverse voorbeelden; parser faalt per document, nooit per batch |
| Chunking te grof/te fijn → slechte retrieval | Middel | chunkparameters config; evaluatieset (commit 10) maakt het meetbaar in plaats van gevoelsmatig |
| LLM negeert bronplicht ondanks prompt | Middel | bronverificatie in code (N2): beweringen zonder geldige ECLI-verwijzing worden afgekeurd |
| Embedding-/LLM-kosten of -afhankelijkheid | Middel | interfaces + lokale defaults; API-backends optioneel via config |
| Scope creep richting UI of heel Rechtspraak | Middel | niet-doelen zijn bindend; subset is afgebakend in OQ3 |

---

## 10. Open vragen

Antwoorden vastleggen in de decisions-log van de README zodra besloten.

1. **Embedding-model:** lokaal (bijv. een meertalig sentence-transformers-model)
   of API? Nederlandstalige juridische tekst is de testcase — timebox in commit 6.
2. **Vector store:** welke embedded store met goede metadatafilters? Beslissen in
   commit 6; de interface uit §5.2 houdt de keuze goedkoop.
3. **Corpus-afbakening:** welke jaren en instanties? Groot genoeg voor zinnige
   retrieval, klein genoeg voor een laptop — beslissen bij commit 3 op basis van
   gemeten volume.
4. **Generatie-LLM:** welk model, en lokaal of API? Kan later dan de rest — de
   retrieval-helft van het systeem is er niet van afhankelijk.

---

## 11. Vaardigheden die dit laat zien (voor CV/site)

RAG-architectuur · chunkingstrategie · embeddings & vector search ·
retrieval-evaluatie (recall@k, bronjuistheid) · hallucinatiebeheersing ·
Python · XML-parsing · CLI-ontwerp · CI met fixtures en fakes
