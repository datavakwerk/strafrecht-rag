# Checklist — strafrecht-rag

**Voortgang: 0 / 10 commits** · repo bevat alleen een README, nog geen code

Bron: [COMMITS.md](COMMITS.md). Vink af bij merge naar `main`, niet bij "werkt
lokaal".

---

## De 10 commits

- [ ] **1 — Project-scaffolding** — repo-init, `.gitignore`, licentie,
      `src/{ingest,index,query}` + `tests/`
- [ ] **2 — Dependencies en tooling** — lockfile, ruff + pytest,
      `make install/lint/test` groen op verse clone
- [ ] **3 — Ingestie** — Rechtspraak.nl-client, XML per ECLI naar `data/raw/`,
      rate limiting + retries, hervatbaar
      *(hier ook: corpus-afbakening beslissen — PRD OQ3)*
- [ ] **4 — Parser** — XML → `Uitspraak`-model, diverse fixtures, per-document
      foutafhandeling *(meest ontwerpzware commit)*
- [ ] **5 — Chunking** — overlap + sectie-/paragraafbehoud, puur en
      deterministisch, config in plaats van constantes
- [ ] **6 — Embeddings + vector store** — achter interfaces, idempotent
      herindexeren, FakeEmbedder in tests
      *(hier: embedding-model en store kiezen — PRD OQ1/OQ2, timeboxen)*
- [ ] **7 — Retrieval** — top-k met filters, score + bronmetadata, scoredrempel
- [ ] **8 — Antwoordgeneratie** — alleen uit passages, ECLI per bewering,
      geen-bron-pad als codepad, bronverificatie in code
      *(het thesis-commit van dit project)*
- [ ] **9 — CLI** — `ingest` / `index` / `ask` end-to-end vanaf de commandline
- [ ] **10 — Evaluatieset + CI** — recall@k en bronjuistheid gemeten (incl.
      geen-bron-vragen), lint + tests op elke push, badge

---

## Na commit 10 (Definition of Done uit de PRD)

- [ ] Corpus opgeschaald naar de volledige afbakening
- [ ] Chunkparameters en drempel getuned op de evaluatiecijfers
- [ ] Recall@k en bronjuistheid in de README, met de vragenset in de repo
- [ ] Decisions-log compleet (embedding-model, store, afbakening, LLM)
- [ ] Disclaimer "geen juridisch advies" in CLI-output en README
- [ ] Repo publiek met CI-badge

---

## Vaste controles per commit

- [ ] `make lint && make test` groen — zonder netwerk, zonder LLM
- [ ] "Klaar wanneer"-criterium uit COMMITS.md gehaald
- [ ] Conventional Commit-bericht zoals in het plan
- [ ] Nieuwe beslissingen genoteerd in de decisions-log
