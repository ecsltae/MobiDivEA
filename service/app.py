"""
Species QA — standalone extraction service.

Given a pre-ingested document (the same JSON shape the ingestion produces, with
`text` and `italicized_terms`), returns the scientific species it contains,
normalized and verified against the local Open Tree Taxonomy index.

No retriever, no LLM: the document is supplied directly (the "doc_ref" is the
document itself). Species are OTT-verified, so misspellings are corrected and
non-taxon italics are filtered out.

Run:
    export SPECIES_QA_OTT_DB=/home/egaillac/MetaP/classifier/data/processed/ott_index.sqlite
    uvicorn app:app --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from extractor import SpeciesExtractor
from location_extractor import LocationExtractor

# Fine-tuned Spanish extractive-QA model (single-answer questions: project
# location, fieldwork dates, process start). Loaded lazily on first /ask call so
# the service starts fast and works even before the model is trained.
_QA_MODEL_DIR = os.environ.get(
    "SPECIES_QA_MODEL_DIR",
    str(Path(__file__).resolve().parents[1] / "qa_bert" / "models" / "boe-qa"),
)

app = FastAPI(
    title="Metadata QA — species & locations",
    description=(
        "Answers two questions about a document, no retriever and no LLM — the "
        "document is supplied directly.\n\n"
        "- **What species?** `/extract-species` — scientific names from the "
        "document's italics (∪ a binomial text scan), verified and normalized "
        "against the local Open Tree Taxonomy (misspellings corrected, "
        "non-taxa filtered, OTT ids attached).\n"
        "- **What locations?** `/extract-locations` — places (municipalities, "
        "rivers, protected areas, …) via Spanish NER with light cleanup.\n\n"
        "Accepts a plain `{id, title, text}` document (e.g. from the triage GUI) "
        "or a full ingestion JSON."
    ),
    version="0.2.0",
    docs_url="/",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Extractors are created lazily and shared across requests. The species one opens
# the OTT index; the location one loads the spaCy Spanish model.
_extractor: Optional[SpeciesExtractor] = None
_loc_extractor: Optional[LocationExtractor] = None


def get_extractor() -> SpeciesExtractor:
    global _extractor
    if _extractor is None:
        _extractor = SpeciesExtractor()
    return _extractor


def get_loc_extractor() -> LocationExtractor:
    global _loc_extractor
    if _loc_extractor is None:
        # Gazetteer index location (mirrors SPECIES_QA_OTT_DB for species).
        _loc_extractor = LocationExtractor(
            geo_db=os.environ.get("SPECIES_QA_GEO_DB")
        )
    return _loc_extractor


_qa = None


def _calibrated_min_score(default: float = 0.0) -> float:
    """No-answer margin threshold. `evaluate_qa.py` picks the F1-optimal value and
    writes it to qa_bert/results/qa_eval.json; use that so the served model doesn't
    over-reject (the fine-tune is no-answer-biased). Override with
    SPECIES_QA_MIN_SCORE."""
    env = os.environ.get("SPECIES_QA_MIN_SCORE")
    if env is not None:
        return float(env)
    ev = Path(__file__).resolve().parents[1] / "qa_bert" / "results" / "qa_eval.json"
    try:
        return float(json.load(open(ev))["best_tau"])
    except Exception:
        return default


def get_qa():
    """Lazily load the fine-tuned QA model (CPU). Imports torch/transformers only
    on first use so the extractor endpoints don't pay for them."""
    global _qa
    if _qa is None:
        qa_dir = str(Path(__file__).resolve().parents[1] / "qa_bert")
        if qa_dir not in sys.path:
            sys.path.insert(0, qa_dir)
        from infer import QA  # noqa: E402
        _qa = QA(_QA_MODEL_DIR, min_score=_calibrated_min_score())
    return _qa


# ── Request / response models ──────────────────────────────────────────────────

class Document(BaseModel):
    """A pre-ingested document. Extra fields are ignored, so the full ingestion
    JSON — or a plain {id, title, text} doc from the triage GUI — can be posted
    verbatim."""
    model_config = {"extra": "allow", "populate_by_name": True}

    id: Optional[str] = Field(default=None, alias="_id")
    text: str = Field(default="", description="Full document text.")
    italicized_terms: List[str] = Field(
        default_factory=list,
        description="Italicised strings from the source (scientific names).",
    )


class ExtractRequest(BaseModel):
    """Post either the whole document under `document`, or `text` +
    `italicized_terms` directly."""
    document: Optional[Document] = None
    text: Optional[str] = None
    italicized_terms: Optional[List[str]] = None
    scan_text: bool = Field(
        default=False,
        description=(
            "Also scan the raw text for non-italicised binomials. "
            "Precision/recall dial measured on the 639 BOE docs vs the is_taxa "
            "flag: false → P=0.99, R=0.85 (curated italics only); "
            "true → P=0.91, R=0.99 (recovers non-italicised species, adds noise)."
        ),
    )
    verified_only: bool = Field(
        default=False,
        description=(
            "Locations only: if true, return just the places confirmed in the "
            "GeoNames gazetteer (canonical name + stable geonameid + admin "
            "context), dropping unverified NER spans. Default false returns all, "
            "with each carrying a `verified` flag."
        ),
    )

    def to_doc(self) -> Dict[str, Any]:
        if self.document is not None:
            d = self.document.model_dump(by_alias=True)
            return d
        return {
            "_id": None,
            "text": self.text or "",
            "italicized_terms": self.italicized_terms or [],
        }


class Mention(BaseModel):
    text: str
    start: Optional[int] = None   # char offset into doc text; None if not located
    end: Optional[int] = None


class SpeciesItem(BaseModel):
    canonical: str
    ott_id: str
    rank: str
    kingdom: Optional[str]
    match_type: str
    score: float
    count: int
    mentions: List[Mention]
    sources: List[str]


class ExtractResponse(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    n_species: int
    species: List[SpeciesItem]

    model_config = {"populate_by_name": True}


class LocationItem(BaseModel):
    name: str          # canonical GeoNames name when verified, else the raw NER span
    type: str          # river | water_body | wetland | relief | pass | protected_area | forest | region | province | comarca | municipality | place
    verified: bool     # confirmed against the GeoNames gazetteer
    geonameid: Optional[str] = None    # stable id, e.g. "geonames:3120514"
    province: Optional[str] = None     # ADM2 context, when known
    region: Optional[str] = None       # ADM1 / comunidad autónoma, when known
    match_type: Optional[str] = None   # exact | core | fuzzy (None if unverified)
    confidence: Optional[float] = None
    count: int
    mentions: List[Mention]


class LocationResponse(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    n_locations: int
    n_verified: int
    locations: List[LocationItem]

    model_config = {"populate_by_name": True}


class AskRequest(BaseModel):
    """Ask one free-text question about a document. Post the whole `document`
    (any ingestion JSON / triage-GUI doc) or just `text`."""
    question: str = Field(description="A single-answer Spanish question, e.g. "
                                      "'¿Cuál es la ubicación del proyecto?'")
    document: Optional[Document] = None
    text: Optional[str] = None

    def context(self) -> str:
        if self.document is not None:
            return self.document.text or ""
        return self.text or ""


class AskResponse(BaseModel):
    question: str
    answer: str                      # "" when the model finds no answer in the doc
    answered: bool
    score: float                     # margin over the model's "no answer" score
    start: Optional[int] = None      # char offset into the document text
    end: Optional[int] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health() -> dict:
    ready = True
    detail = "ok"
    try:
        get_extractor()
    except Exception as e:  # OTT index missing / not built yet
        ready, detail = False, str(e)
    return {"status": "healthy" if ready else "degraded", "ready": ready, "detail": detail}


@app.post("/extract-locations", response_model=LocationResponse, tags=["Locations"])
def extract_locations(req: ExtractRequest) -> dict:
    """Answer "What locations?" — places from Spanish NER, verified/normalized
    against the local GeoNames gazetteer (canonical name, stable geonameid, type,
    admin context). Set `verified_only` to drop unconfirmed spans."""
    return get_loc_extractor().extract(req.to_doc(), verified_only=req.verified_only)


@app.post("/extract-species", response_model=ExtractResponse, tags=["Species"])
def extract_species(req: ExtractRequest) -> dict:
    """Extract OTT-verified species from a single document."""
    ex = get_extractor()
    return ex.extract(req.to_doc(), scan_text=req.scan_text)


@app.post("/extract-species/batch", tags=["Species"])
def extract_species_batch(docs: List[Document]) -> dict:
    """Extract species from several documents in one call."""
    ex = get_extractor()
    results = [ex.extract(d.model_dump(by_alias=True)) for d in docs]
    return {"results": results, "count": len(results)}


@app.post("/ask", response_model=AskResponse, tags=["QA"])
def ask(req: AskRequest) -> dict:
    """Answer a single-answer question about a document with the fine-tuned Spanish
    extractive-QA model. Returns the best text span + char offsets, or an empty
    answer when the model finds none in the document."""
    res = get_qa().answer(req.question, req.context())
    return {
        "question": req.question,
        "answer": res["answer"],
        "answered": bool(res["answer"]),
        "score": res["score"],
        "start": res["start"],
        "end": res["end"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8010)
