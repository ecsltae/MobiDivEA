# Metadata QA — species, locations & free-text QA service

Answers questions about a document — **"What species?"**, **"What locations?"**,
and single-answer free-text questions (**"¿Cuál es la ubicación del proyecto?"**)
— with no retriever (the document is supplied directly, so the "doc_ref" is the
document itself). Accepts a plain `{id, title, text}` doc (e.g. straight from the
triage GUI) or a full ingestion JSON.

- **Species** (`/extract-species`) — verified and normalized against the local
  Open Tree Taxonomy (misspellings corrected, non-taxa filtered, OTT ids
  attached). Deterministic, no model.
- **Locations** (`/extract-locations`) — Spanish NER (spaCy `es_core_news_md`)
  for recognition, then **verified/normalized against a local GeoNames Spain
  gazetteer** (canonical name, stable `geonameid`, accurate type, province/region
  context) — the location analogue of OTT for species. Deterministic, no model.
- **Free-text QA** (`/ask`) — a fine-tuned Spanish extractive-QA model for
  single-answer questions the two list-style extractors can't handle (project
  location, fieldwork dates, process start date, ...). See below.

Built for Maite's 600-document metadata task and designed to sit next to the
MobiDivEA triage `/classify` API (which classifies P/S but does not enumerate taxa).

## Shareable URL

Bound to `0.0.0.0`, so once running it is reachable on the LAN at:

```
http://egaillac.lan.text-analytics.ch:8010      (docs at /)
```

## Why species/locations are rule-based, not a trained model

The input is *pre-ingested* JSON in which scientific names are already italicised
(`italicized_terms`), and we only need species. So the reliable, explainable
approach is **candidate generation + OTT verification**, not a neural net trained
on ~421 weakly-labelled docs:

1. **Candidates** — `italicized_terms` (∪ an optional conservative binomial scan
   of `text`).
2. **Verify + normalize** — each candidate is resolved through the local OTT index
   (`ott_resolver.py`): exact → synonym → abbreviated-genus → fuzzy. This *is* the
   "is it really a species?" filter — legal-Latin / emphasis italics (`in situ`,
   `Boletín Oficial`) don't resolve; real taxa do, and misspellings are corrected
   (`Olea europea` → `Olea europaea`).
3. **Dedupe** — group by OTT id; keep provenance (raw variants, italics vs text,
   mention count).

## Validation (639 BOE docs, vs the `is_taxa` gold flag)

| Mode | Precision | Recall | Notes |
|---|---|---|---|
| italics-only (`scan_text=false`, default) | **0.989** | 0.854 | 7,355 species; misses non-italicised taxa |
| + text binomial scan (`scan_text=true`) | 0.914 | **0.993** | recovers non-italicised species, adds noise |

Match types over the italics path: 6,865 exact · 444 synonym · 45 fuzzy · 1 abbrev.
The 45 fuzzy hits are genuine misspelling corrections (`Aythys nyroca`→`Aythya
nyroca`, `Egretta garceta`→`garzetta`, …). Two of four apparent false positives
were real species the gold flag had missed. ~90 ms/doc, single-threaded.

## Free-text QA (`/ask`)

Species/locations are list-style extractors; some questions need one answer
picked out of running text (*"¿Cuál es la ubicación del proyecto?"*, *"¿Cuándo
se realizó el trabajo de campo?"*). For those, `/ask` runs a fine-tuned Spanish
**extractive QA** model (SQuAD-v2 style: a text span, or "no answer" when the
document doesn't contain one) over the document, no retriever.

- **Base model**: `mrm8488/roberta-base-bne-finetuned-sqac` (RoBERTa-BNE,
  pre-tuned on the Spanish QA Corpus), domain-adapted on the BOE docs.
- **Training data** (`../qa_bert/prepare_data.py`, SQuAD-v2 format): the MobiDivEA
  triage `/classify` evidence spans (theme → question, e.g. `wind_farm` → *"¿Qué tipo de
  proyecto?"*), our species/location extractor mentions (entity-question spans),
  and an LLM-teacher pass over `../qa_bert/questions.txt` for the open,
  single-answer questions. No human-labelled QA pairs.
- **Serving**: trained on GPU (`../qa_bert/train_qa.py`), served on CPU
  (`../qa_bert/infer.py`, sliding-window decode for long docs), loaded lazily on
  first `/ask` call. Current model: `../qa_bert/models/boe-qa-v2`.
- **No-answer threshold**: calibrated per-model by `../qa_bert/evaluate_qa.py`
  (F1-optimal `best_tau` over a held-out dev split), written to
  `../qa_bert/results/qa_eval.json` and read automatically at startup — override
  with `SPECIES_QA_MIN_SCORE`. **If you retrain, re-run `evaluate_qa.py` against
  the new model and point `SPECIES_QA_MODEL_DIR` (env var, also set in
  `species-qa.service`) at it — the threshold and the model must match.**
- **Eval caveat**: `qa_eval.json` is *teacher-agreement*, not human-gold (dev set
  is held-out docs scored against the same LLM-teacher labels used for training,
  not a separately annotated ground truth). Treat scores as a sanity check, not a
  publishable benchmark.

## Prerequisite: build the OTT index (one-off)

The resolver (`ott_resolver.py`, vendored in this directory, stdlib-only) reads a
SQLite index built from an Open Tree of Life taxonomy dump. That dump and the
built index are **not in this repo** (2.4 GB source JSON → 1.0 GB SQLite index,
4.53M concepts / 9.3M names) — not excluded by accident, they're just too big to
version. On `egaillac`'s machine the built index already exists at
`/home/egaillac/MetaP/classifier/data/processed/ott_index.sqlite`; point
`SPECIES_QA_OTT_DB` at it directly, or rebuild elsewhere:

```bash
# 1. get an OTT taxonomy dump (https://tree.opentreeoflife.org/about/taxonomy-version)
#    as JSON, e.g. ott_v3.7.2.json
# 2. build the index
python ott_resolver.py build --json /path/to/ott_v3.7.2.json --db ott_index.sqlite
```

## Prerequisite: build the locations gazetteer (one-off)

The location verifier (`geonames_resolver.py`, vendored here, stdlib-only + optional
rapidfuzz) reads a SQLite index built from the GeoNames **Spain** dump. The dump and
the built index are **not in this repo** (~3 MB `ES.txt` → ~16 MB index, 57,673
places) — regenerate with the free download:

```bash
curl -O https://download.geonames.org/export/dump/ES.zip && unzip ES.zip   # → ES.txt
python geonames_resolver.py build --txt ES.txt --db geonames_index.sqlite
```

Only geographic feature classes are indexed (admin, populated places, hydrography,
relief, parks, forest); GeoNames class S (hotels/museums/…) and R (roads) are
excluded on purpose. If the index is absent the service still runs — locations just
come back `verified:false`.

## Run

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_md          # locations NER
export SPECIES_QA_OTT_DB=/home/egaillac/MetaP/classifier/data/processed/ott_index.sqlite
export SPECIES_QA_GEO_DB=/home/egaillac/MetaP/classifier/data/processed/geonames_index.sqlite
./start.sh                                         # binds 0.0.0.0:8010
```

### Persistent deployment (for a durable shareable URL)

Run it under something that outlives your shell so the triage GUI can rely on it:

```bash
# systemd (recommended; needs sudo once)
sudo cp species-qa.service /etc/systemd/system/
sudo systemctl enable --now species-qa
# or, quick and dirty, in a detached session:
tmux new -d -s speciesqa '/home/egaillac/MetaP/species_qa_service/start.sh'
```

Then confirm from another host: `curl http://egaillac.lan.text-analytics.ch:8010/health`.
If it is not reachable from the triage host, open TCP 8010 in the firewall.

`species-qa.service` already pins `SPECIES_QA_MODEL_DIR` at the current best QA
checkpoint (`qa_bert/models/boe-qa-v2`) — after retraining, update that line (and
re-run `evaluate_qa.py`, see [Free-text QA](#free-text-qa-ask)) before
re-installing the unit, or the service will keep serving the old checkpoint.

## API

### `POST /extract-species`
Post the whole ingestion JSON under `document` (extra fields ignored), or `text` +
`italicized_terms` directly.

```jsonc
// request
{ "document": { "_id": "BOE-A-2015-2023", "text": "...", "italicized_terms": ["Olea europea", "in situ"] },
  "scan_text": false }

// response
{ "_id": "BOE-A-2015-2023", "n_species": 1,
  "species": [
    { "canonical": "Olea europaea", "ott_id": "ott:23729", "rank": "species",
      "kingdom": "Chloroplastida", "match_type": "fuzzy", "score": 0.96,
      "count": 1, "mentions": ["Olea europea"], "sources": ["italics"] }
  ] }
```

### `POST /extract-locations`
Same request shape. Answers *"What locations?"* via Spanish NER, then verifies
each span against the GeoNames gazetteer. Optional `verified_only` (default false)
drops spans that don't resolve.

```jsonc
// request
{ "document": { "_id": "BOE-A-2020-5107", "text": "..." }, "verified_only": true }

// response
{ "_id": "BOE-A-2020-5107", "n_locations": 14, "n_verified": 14,
  "locations": [
    { "name": "Río Cinca", "type": "river", "verified": true,
      "geonameid": "geonames:3125022", "province": null, "region": null,
      "match_type": "exact", "confidence": 1.0, "count": 2,
      "mentions": [ { "text": "río Cinca", "start": 1234, "end": 1243 } ] },
    { "name": "Provincia de Huesca", "type": "province", "verified": true,
      "geonameid": "geonames:3120513", "province": "Provincia de Huesca",
      "region": "Aragón", "match_type": "exact", "confidence": 1.0, "count": 4,
      "mentions": [ ... ] }
  ] }
```

`type` ∈ river · water_body · wetland · relief · pass · protected_area · forest ·
region · province · comarca · municipality · place. `match_type` ∈ exact · core
(head-noun stripped, e.g. "río X"→"X"). Unverified spans (default response) carry
`verified:false` and null gazetteer fields. First location request loads the
spaCy model (~1–2 s).

**Quality (audited, no human gold — LLM-judged against document context over a
60-doc sample):** of the NER spans that survive pre-filtering, ~22% verify; the
**verified list is ~75% precise** (right place, right type) and the **unverified
bucket is ~80–85% genuine noise** (taxa/habitat names spaCy mislabels as places,
government bodies, fragments). `verified_only=true` is the clean list; the default
keeps everything with a `verified` flag so no real place is silently dropped.
Residual verified errors are mostly cross-province same-name collisions (e.g.
"Sella" = an Asturias river *and* an Alicante town); recall gaps are places
GeoNames Spain lacks (micro-toponyms, small *arroyos*, *vías pecuarias*, truncated
protected-area names, foreign places). See Roadmap for the geographic-coherence
fix.

### `POST /extract-species/batch`
Body: a JSON list of documents. Returns `{ "results": [...], "count": N }`.

### `POST /ask`
Post `document` (or `text`) + a single-answer `question`. Answers with the
fine-tuned Spanish extractive-QA model (see [Free-text QA](#free-text-qa-ask)
above); no retrieval, the document is the whole context.

```jsonc
// request
{ "document": { "_id": "BOE-A-2011-11006", "text": "..." },
  "question": "¿Cuál es la ubicación del proyecto?" }

// response
{ "question": "¿Cuál es la ubicación del proyecto?",
  "answer": "El helipuerto se emplaza en el término municipal de Villaeles de Valdavia (provincia de Palencia, Castilla y León).",
  "answered": true, "score": 3.48, "start": 4763, "end": 4878 }
```

`answered:false` (empty `answer`, `start`/`end` null) means the model's
no-answer score beat every span in the document — normal for questions that
don't apply to that doc (e.g. asking fieldwork dates on an admin-only filing).

### `GET /health`
`ready:false` with a `detail` message if the OTT index is missing.

## GUI integration

The GUI holds the document (it produced/ingested it), so it POSTs it straight to
`/extract-species` and renders `species[]` — no retrieval round-trip. Contract:

- **canonical / ott_id** — the display name and the stable id (use `ott_id` as the
  key to link, dedupe, or cross-reference; same OTT backbone as the thesis
  re-ranker).
- **match_type / score** — surface `fuzzy` hits for optional reviewer
  confirmation; `exact`/`synonym` are safe to show directly.
- **mentions** — the raw strings as they appear in the document (for highlighting).
- **scan_text** — leave `false` for curated-italics precision; set `true` when the
  GUI needs to catch species that were not italicised (accepts more noise).

CORS is open; deploy behind the same reverse proxy as the other SIBiLS services if
a shared origin is required.

## Roadmap

- **Location geographic coherence** — the GeoNames gazetteer verifier (done)
  lifted the verified list to ~75% precision; the remaining errors are
  cross-province same-name collisions. Next: anchor on a document's unambiguous
  province/region hits and prefer same-province candidates for the ambiguous ones.
  Also consider `es_core_news_lg` / GLiNER for recognition, and cross-referencing
  the species list to drop taxon leakage ("Olea" the genus vs the hamlet).
- **Common-name species** — docs that name taxa only by Spanish common name
  (e.g. "milano", "aves esteparias") resolve to nothing; add a Spanish
  common-name → taxon gazetteer.
- **Recognition upgrade** — for raw text, swap the binomial regex for gnfinder as
  the candidate generator, still resolving against the local OTT. See
  `../thesis_docs/gnfinder_assessment.md`.
