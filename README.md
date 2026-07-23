# MobiDivEA — Maite metadata QA

Metadata extraction over ~600 Spanish BOE (*Boletín Oficial del Estado*)
environmental-impact gazette documents, built for Maite's document-review task:
**what species, what locations, and single-answer facts** (project location,
fieldwork dates, ...) does each document mention. Designed to sit next to 
MobiDivEA triage system (`/classify`, Primary/Secondary evidence scoring) — that
system classifies documents but does not enumerate taxa or answer free-text
questions; this one does.

## Layout

```
sample_revised/          639 source documents (JSON: text, italicized_terms, ...)
training_QA-maite.csv    doc-level gold flags: is_taxa, is_location, is_primary_data,
                         is_secondary_data (is_location is degenerate — all 1)
service/                 FastAPI service — the deployed deliverable
  app.py                   /extract-species, /extract-locations, /ask, /health
  extractor.py             species: candidates (italics ∪ text scan) → OTT-verify
  location_extractor.py    locations: localización-section NER → GeoNames verify
  ott_resolver.py          vendored OTT name resolver (species, stdlib-only)
  geonames_resolver.py     vendored GeoNames Spain gazetteer resolver (locations)
  vernacular_resolver.py   Spanish common-name → scientific detector (species)
  vernacular_es.tsv        common-name gazetteer (Wikidata-derived, ~24k names)
  build_vernacular.py      rebuild vernacular_es.tsv from a Wikidata dump
  README.md                full API docs, validation numbers, deployment steps
qa_bert/                 training pipeline for the /ask model (not run automatically)
  PLAN.md                  design notes: why extractive QA, data sources, model choice
  prepare_data.py          builds SQuAD-v2 training data (triage evidence spans +
                           our extractors + LLM-teacher labels over questions.txt)
  train_qa.py              GPU fine-tune (HF AutoModelForQuestionAnswering)
  evaluate_qa.py           dev-set eval + no-answer threshold calibration
  infer.py                 CPU inference, sliding-window decode
  questions.txt            the example questions the trained model answers
  results/                 small eval summaries (qa_eval.json, qa_eval_v2.json)
```

Start with `service/README.md` for how to run the service and its full API, or
`qa_bert/PLAN.md` for how the free-text QA model was trained.

## What's *not* in this repo

Large, regeneratable artifacts were deliberately left out (see `.gitignore`):

- **`qa_bert/data/`** — generated SQuAD-v2 training data (`squad_train.json` ~200MB,
  `squad_dev.json` ~21MB) and LLM-teacher labels (`teacher.jsonl` ~150MB). Rebuild
  with `qa_bert/prepare_data.py`.
- **`qa_bert/models/`** — trained checkpoints (hundreds of MB–GBs each). Rebuild
  with `qa_bert/train_qa.py` (GPU). The currently-deployed checkpoint is
  `boe-qa-v2`.
- **OTT taxonomy dump + built SQLite index** (2.4GB source JSON → 1.0GB index) that
  `service/ott_resolver.py` needs for species verification. See
  `service/README.md` → "Prerequisite: build the OTT index".
- **GeoNames Spain gazetteer + built SQLite index** (~3MB `ES.txt` → ~16MB index)
  that `service/geonames_resolver.py` needs for location verification. Free
  download; see `service/README.md` → "Prerequisite: build the locations gazetteer".

## Current deployment

This repo is the source-controlled copy of the code. The live instance actually
serving requests runs from `/home/egaillac/MetaP/species_qa_service` +
`/home/egaillac/MetaP/qa_bert` on host `egaillac.lan.text-analytics.ch`
(systemd unit `species-qa`, enabled — survives reboot), reachable at:

```
http://egaillac.lan.text-analytics.ch:8010      (API docs at /)
```

If you want this repo's checkout to *become* the deployment source of truth,
update the paths in `service/species-qa.service` (`WorkingDirectory`,
`ExecStart`, and the `SPECIES_QA_OTT_DB` / `SPECIES_QA_GEO_DB` /
`SPECIES_QA_MODEL_DIR` env vars) to point here, then reinstall the unit as
described in `service/README.md`.
