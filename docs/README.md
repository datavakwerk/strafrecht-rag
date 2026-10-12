# Docs — strafrecht-rag

Planningsdocumenten voor **strafrecht-rag**: een RAG-systeem over Nederlandse
rechterlijke uitspraken (Rechtspraak.nl Open Data) — semantisch zoeken, antwoorden
met verplichte ECLI-bronvermelding, en zonder bron géén antwoord. Deze map staat
bewust **buiten** de repo — werkplan, geen deliverable.

**Repo:** `../../strafrecht-rag/` (alleen README — nog geen code)
**Status:** niet gestart — 0 van 10 commits.

| Document | Wat |
|---|---|
| [PRD.md](PRD.md) | Doelen, architectuur, bronplicht-ontwerp, Definition of Done, open vragen |
| [COMMITS.md](COMMITS.md) | De eerste 10 commits: van scaffold tot evaluatieset + CI |
| [checklist.md](checklist.md) | Voortgang per commit plus het werk ná commit 10 |

## Werkwijze

Eén commit per sitting, repo blijft groen, fixture-first (geen netwerk of LLM in
CI). De grote keuzes — embedding-model, vector store, corpus-afbakening — zijn
bewust naar commits 3 en 6 geschoven, achter interfaces; zie de open vragen in
PRD §10 en de beslissingentabel onderin COMMITS.md.
