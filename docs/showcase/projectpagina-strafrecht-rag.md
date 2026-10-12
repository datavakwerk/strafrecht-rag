# strafrecht-rag

**RAG over 1.906 Nederlandse strafrechtuitspraken, met één regel: geen bron, geen antwoord. Elke bewering eindigt in een ECLI, sectie en paragraaf — en die verwijzing wordt in code gecontroleerd, niet aan het model gevraagd.**

---

## Hero / intro

Een RAG-demo bouwen is een middag werk: documenten inlezen, embeddings maken, top-k ophalen, in een prompt plakken. Het probleem zit niet in de pipeline maar in wat er uit komt. Een taalmodel dat vier passages krijgt en gevraagd wordt om een antwoord met bronnen, produceert vrijwel altijd iets dat er correct uitziet. Een ECLI die net niet bestaat, een paragraafnummer dat nergens op slaat, een bewering die klopt maar niet in de aangeleverde tekst staat — je ziet het verschil niet zonder het na te lopen.

In het strafrecht is dat verschil het hele punt. Dus is dit project omgedraaid: niet "hoe krijg ik het model zover dat het netjes citeert", maar "wat kan ik na afloop hard controleren, en wat gooi ik weg als het niet klopt". Een bewering zonder verwijzing haalt het antwoord niet. Een ECLI die niet in de opgehaalde passages voorkomt haalt het antwoord niet. En als de beste passage onder de relevantiedrempel blijft, wordt het model helemaal niet aangeroepen.

*Korte feiten voor onder de intro:* Python 3.12 · SQLite + NumPy · FTS5/BM25 · fastembed (ONNX) · cross-encoder reranking · Ollama / Claude / extractief — 1.906 uitspraken, 124.952 chunks, 93 tests, ruff schoon — draait volledig lokaal, zonder account of sleutel.

---

## Waarom dit project

Ik wilde weten waar de grens ligt tussen "een RAG-systeem dat demonstreert" en "een RAG-systeem waarvan je de uitvoer durft door te sturen". Die grens ligt niet bij het ophaalgedeelte — daar zijn de recepten bekend. Die ligt bij de vraag wat er gebeurt als het model iets zegt wat niet uit de bronnen komt.

Rechtspraak.nl publiceert alle uitspraken als open data, geanonimiseerd, met een gestructureerde XML. Dat is een corpus waar de antwoorden objectief verifieerbaar zijn: een ECLI bestaat of bestaat niet, een paragraaf staat er of staat er niet. Precies het soort domein waar je een controlemechanisme kunt bouwen dat meer is dan een gevoel.

Alleen de gepubliceerde, geanonimiseerde tekst wordt gebruikt. Er wordt niets verrijkt en niets gede-anonimiseerd.

---

## Architectuur

> **[DIAGRAM 1 — rag-1-architectuur]**

Het systeem valt uiteen in twee helften die elkaar alleen via de index raken.

De **offline helft** draait je één keer per corpus-uitbreiding. Een client bevraagt de Open Data zoekfeed op datum en rechtsgebied, pagineert door de resultaten en haalt per ECLI één XML op naar `data/raw/`. Rate limited, met backoff, en herstartbaar: bestaande bestanden worden overgeslagen. De parser maakt daar één `Uitspraak` van — instantie, datum, rechtsgebieden, procedures, zaaknummer, wetsverwijzingen, de redactionele samenvatting, en genummerde secties met genummerde paragrafen. De chunker knipt die in stukken die nooit een sectiegrens oversteken, en de embedder zet ze om in vectoren.

De **online helft** is één vraag lang. De retriever haalt kandidaten op, de cross-encoder herordent ze en geeft er een gekalibreerde score aan, de drempel bepaalt of het model überhaupt wordt aangeroepen, en de verificatie bepaalt wat er van het antwoord overblijft.

Daartussen zit één opslaglaag: SQLite met de metadata en een FTS5-index, plus de vectoren als één float32-matrix in NumPy. Eén bestand, geen service.

---

## De ene regel: geen bron, geen antwoord

> **[DIAGRAM 2 — rag-2-geen-bron-geen-antwoord]**

De regel wordt op drie plaatsen afgedwongen, en op geen van die plaatsen door de prompt.

**Voor generatie.** Ligt de beste relevantiescore onder de drempel (0,35), dan wordt de LLM niet aangeroepen. "Geen bron" is daarmee een eigenschap van het ophalen, niet een beleefdheid van het model.

**Tijdens generatie.** Het model krijgt genummerde passages en de opdracht om per regel één bewering te geven die eindigt op een `[n]`-marker. Het mag ook `GEEN_BRON` antwoorden als het vindt dat de passages de vraag niet dekken.

**Na generatie.** Elke bewering gaat langs een keten van controles: staat er een marker? Wijst die marker naar een passage die daadwerkelijk is aangeleverd? Komt elke ECLI in de bewering voor in die passages? Is de bewering niet leeg en niet al eerder gezegd? Wat afvalt, valt af — met de reden erbij, zichtbaar in de uitvoer. Blijft er niets over, dan krijgt het model één herkansing met een formatherinnering; wat het de eerste keer produceerde blijft bewaard in de ruwe output.

Dat laatste is het punt waar de meeste RAG-systemen stoppen bij een instructie in de systeemprompt. Een instructie is geen garantie. De testsuite bevat een nep-LLM die een ECLI verzint en een die de verwijzingen weglaat; beide worden afgekeurd. En de evaluatieset bevat vragen waarvan het antwoord *moet* zijn dat het corpus geen bron heeft — dat pad wordt dus gemeten, niet aangenomen.

De opbrengst staat in één getal: **claimacceptatie 89%**. Elf procent van wat het model produceerde week af van de passages en heeft de gebruiker nooit bereikt.

---

## Waarom er een cross-encoder in zit

Dit is de meting die het ontwerp bepaalde.

Het voor de hand liggende recept is: cosine-similariteit, drempel eroverheen, onder de drempel zeg je "geen bron". Dat werkt niet. De cosine van een bi-encoder is niet gekalibreerd — een vraag die volstrekt niets met het corpus te maken heeft scoort nog steeds rond de 0,6 tegen wíllekeurig welke juridische paragraaf, simpelweg omdat juridisch Nederlands op juridisch Nederlands lijkt. Er is geen drempelwaarde die "relevant" van "toevallig hetzelfde register" scheidt.

Een cross-encoder ziet vraag en passage samen en produceert wél een bruikbare relevantiekans. Dat is duurder — maar alleen over de top-12 kandidaten, en dat kost een paar seconden op een laptop-CPU. De drempel van 0,35 ligt op dat getal, niet op de cosine.

Het ophalen zelf is hybride: cosine top-3k en BM25 top-3k, samengevoegd met reciprocal rank fusion (`k=60`). Dat is geen theoretische voorkeur. Juridisch Nederlands is exact — ECLI-nummers, artikelnummers, plaatsnamen, `artikel 310 Sr` — en daar is een kleine multilinguale embedder slecht in en BM25 juist goed. Daarbovenop mogen er hoogstens twee chunks per uitspraak in de top-k, zodat één breedsprakige uitspraak de hele context niet opslokt.

---

## Van XML naar een verwijzing

> **[DIAGRAM 3 — rag-3-van-xml-naar-verwijzing]**

De verwijzing die de gebruiker ziet — *ECLI:NL:RBNHO:2024:1234, §4.2 Waardering van het bewijs › Het oordeel van de rechtbank, par. 3* — moet ergens vandaan komen, en dat is het onopvallende deel van het project waar het meeste werk in zit.

Uitspraken zetten hun structuur niet netjes in tags. *Standpunt van de verdediging* en *Oordeel van de rechtbank* staan vaak als een losse vetgedrukte alinea midden in een sectie, niet als een eigen `<section>`. De parser splitst daarop: een alleenstaande vetgedrukte of onderstreepte alinea maakt een subsectie. Dat maakt zowel de verwijzing preciezer als de embedding, omdat een chunk dan niet meer twee tegengestelde standpunten in één vector propt.

Het XML-schema varieert bovendien per jaar en per instantie. De parser matcht daarom op lokale elementnamen zonder namespace-aannames, en wordt getest tegen opgenomen echte uitspraken van zowel een rechtbank als een hof. Eén onparsebaar bestand levert één foutrecord op, nooit een gecrashte batch. Over 1.906 uitspraken: **0 parse-fouten**.

De chunker is puur en deterministisch: dezelfde uitspraak geeft altijd dezelfde chunks met dezelfde id's, wat betekent dat opnieuw indexeren geen duplicaten oplevert. Chunks steken nooit een sectiegrens over en overlappen in hele paragrafen, zodat de paragraafnummers in de verwijzing blijven kloppen.

De maatvoering — 560 tekens met 120 overlap — komt uit het model. De embedder ziet maximaal 128 tokens, ongeveer 450 tekens. Een chunk van venster plus overlap betekent dat het stuk dat voorbij het venster valt terugkomt als begin van de volgende chunk, en dus alsnog geëmbed wordt. BM25 ziet ondertussen wél elk teken, want die kijkt naar de volledige tekst in FTS5.

---

## Het corpus

Strafrechtuitspraken met een volledige tekst, over de eerste helft van maart in drie jaren (2023, 2024, 2025). Groot genoeg voor echt ophaalgedrag, klein genoeg om op een laptop te bouwen.

| | |
|---|---|
| Uitspraken | 1.906 (1.245 rechtbank, 416 gerechtshof, 132 Hoge Raad, 89 Parket bij de Hoge Raad, 24 overig) |
| Parse-fouten | 0 |
| Chunks | 124.952 |
| Instanties | 22 |
| Indexomvang | 473 MB SQLite |

Drie jaren in plaats van één, omdat een datumfilter pas iets betekent als er wat te filteren valt. Alle instanties, omdat het type-filter (rechtbank vs. hof) anders leeg is.

---

## Evaluatie

Vijftien vragen in `tests/fixtures/eval_vragen.json`: acht semantische (bewust anders geformuleerd dan de uitspraken zelf), twee met metadatafilters, twee over de inhoud van één specifieke uitspraak, en drie waarop het corpus geen antwoord heeft.

| Metriek | Waarde | Betekenis |
|---|---|---|
| recall@6 | **93%** | aandeel verwachte ECLI's in de top-k |
| hit@6 | 100% | minstens één verwachte ECLI in de top-k |
| MRR | 0,94 | positie van de eerste verwachte ECLI |
| bronjuistheid | **100%** | antwoorden waarvan elke ECLI uit de opgehaalde passages komt |
| claimacceptatie | 89% | ruwe modelbeweringen die de verificatie overleven |
| geen-bron-correctheid | **100%** | onbeantwoordbare vragen die het geen-bron-pad halen |
| onterecht geen bron | 0% | beantwoordbare vragen die geweigerd worden |

Bronjuistheid van 100% is geen prestatie maar een constructie: een bewering met een onbekende ECLI komt het antwoord niet in. Het interessante getal is claimacceptatie, want dat laat zien hoe vaak het model wél van de passages af zou zijn gedwaald als niemand had gekeken.

De ophaalmetrieken hebben geen LLM nodig. De generatiemetrieken zijn gemeten met `ollama/llama3.2`, lokaal.

---

## Ontwerpkeuzes

**SQLite + NumPy in plaats van een vectordatabase.** Bij tienduizenden vectoren is een exacte scan een handvol milliseconden. FTS5 geeft BM25 er gratis bij, het geheel is één bestand, er draait geen service naast, en er zijn geen platformspecifieke wheels nodig — wat concreet uitmaakte, want LanceDB en recente onnxruntime hebben geen Intel-Mac wheels. Een ANN-index zou hier snelheid inruilen voor exactheid zonder dat er snelheid te winnen valt.

**Een lokale, gekwantiseerde ONNX-embedder.** `paraphrase-multilingual-MiniLM-L12-v2` via fastembed: 118M parameters, ongeveer 13 chunks per seconde op een laptop-CPU. Grotere modellen (e5-large, bge-m3) waren op deze machine een orde van grootte trager voor winst die de evaluatie niet liet zien. Het zit achter een `Embedder`-interface, dus wisselen is configuratie.

**Drie LLM-backends, waarvan één zonder model.** Ollama is de standaard (geen sleutel nodig), Claude is optioneel, en er is een extractieve backend die simpelweg de beste passages letterlijk citeert. Die derde is er niet voor de show: daarmee draait de hele keten inclusief evaluatie zonder dat er ergens een model geladen hoeft te worden. En hij gaat door dezelfde verificatie heen als de andere twee.

**Alles achter een interface, alles nep-baar in tests.** `FakeEmbedder` (hashed bag-of-words), `FakeLLM`, `FakeReranker`, opgenomen XML-fixtures. 93 tests, en de CI draait zonder netwerk, zonder modeldownload en zonder LLM. Dat is geen zuiverheidsprincipe maar snelheid: de testsuite is klaar voordat je van tabblad bent gewisseld.

---

## Wat dit project laat zien

Dat de moeilijke vraag bij RAG niet "hoe haal ik de juiste stukken op" is, maar "wat doe ik als het model iets zegt wat er niet staat". Ophalen is een oplosbaar probleem met bekende recepten. Verifiëren is een ontwerpkeuze die je bewust moet maken, en die je vervolgens moet meten — anders weet je niet of hij werkt.

Verder: dat een gekalibreerde score iets fundamenteel anders is dan een similariteitsgetal, dat een parser die het echte, rommelige schema aankan meer waard is dan een parser die de documentatie volgt, en dat "het draait lokaal, zonder account, in één bestand" een architectuurkeuze is en geen beperking.

---

## Disclaimer

Geen juridisch advies. Dit is een zoek- en samenvattingshulpmiddel over gepubliceerde, geanonimiseerde uitspraken; controleer elke bron via de ECLI op rechtspraak.nl.

---

## Links

- Repository: [github.com/datavakwerk/strafrecht-rag](https://github.com/datavakwerk/strafrecht-rag)
- Databron: [Rechtspraak.nl Open Data](https://www.rechtspraak.nl/Uitspraken/paginas/open-data.aspx)

---

### Diagrammen in dit artikel

| # | Bestand | Onderwerp |
|---|---|---|
| 1 | `rag-1-architectuur.drawio` | Offline indexering en online beantwoording, met de index ertussen |
| 2 | `rag-2-geen-bron-geen-antwoord.drawio` | De verificatieketen per bewering en de twee manieren om "geen bron" te zeggen |
| 3 | `rag-3-van-xml-naar-verwijzing.drawio` | Van ruwe XML via secties en chunks naar de verwijzing die de gebruiker ziet |
