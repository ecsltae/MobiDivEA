"""
location_extractor.py — location extraction from a Spanish (BOE) document.

Multilingual NER (spaCy `es_core_news_md`) for *recognition*, then a GeoNames
gazetteer for *verification* — mirroring the species side (NER/candidates → OTT).
There are NO gold location labels, so this stays unsupervised:

    1. recognition   spaCy proposes LOC spans.
    2. pre-filter    drop administrative/legal noise, digit/parcel codes, stubs
                     (cheap rules; also catches govt-body words like "Ministerio"
                     that happen to collide with tiny hamlet names in GeoNames).
    3. verify        resolve each span against the local GeoNames Spain gazetteer
                     (`geonames_resolver.py`): confirmed places get a canonical
                     name, a stable `geonameid`, an accurate `type` from the
                     feature code, and admin context (province / comunidad).
                     Unconfirmed spans are kept but flagged `verified=false`
                     (no recall loss) — or dropped entirely with `verified_only`.
    4. dedupe        by geonameid when verified, else by normalized name.

The gazetteer is the location analogue of OTT for species. If the index isn't
built/available, the extractor degrades gracefully to recognition + pre-filter
(every location comes back `verified=false`).
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Vendored gazetteer resolver, alongside this file.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
try:
    from geonames_resolver import GeoNamesResolver  # noqa: E402
except Exception:  # pragma: no cover
    GeoNamesResolver = None  # type: ignore


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


# Entities spaCy tags as LOC but which are administrative bodies / legal refs, not
# places. Matched as whole-word tokens on the accent-stripped entity text.
_ADMIN_STOP = {
    "ley", "decreto", "orden", "reglamento", "articulo", "capitulo", "titulo",
    "seccion", "anexo", "expediente", "boletin", "gobierno", "ministerio",
    "departamento", "consejeria", "direccion", "subdireccion", "servicio",
    "oficina", "secretaria", "delegacion", "ayuntamiento", "diputacion",
    "sostenibilidad", "ganaderia", "demografico", "tramo", "plan", "programa",
    "resolucion", "disposicion", "fundamentos", "derecho", "objeto",
    "instituto", "confederacion", "sociedad", "gasoducto", "oleoducto",
    "carretera", "autovia", "autopista", "sau", "sa",
}

# Leading geographic head-noun → coarse type. Order matters (first match wins).
_TYPE_HINTS = [
    ("river",         {"rio", "arroyo", "barranco", "rambla", "ribera"}),
    ("water_body",    {"embalse", "laguna", "pantano", "charca", "humedal", "balsa"}),
    ("relief",        {"sierra", "monte", "montes", "puerto", "valle", "pico", "collado", "cerro"}),
    ("protected_area",{"parque", "reserva", "zepa", "lic", "zec", "espacio", "paraje", "monumento"}),
    ("trail",         {"canada", "vereda", "cordel", "colada", "via"}),
    ("province",      {"provincia"}),
    ("municipality",  {"termino", "municipio"}),
]

# Fragments we never want as a location: pure punctuation/digits or 1-2 chars.
_JUNK = re.compile(r"^[\W\d]+$|^.{1,2}$", re.IGNORECASE)
# Real place names in this corpus carry no digits; road/parcel codes ("A3.4",
# "2000") do — reject anything containing a digit anywhere.
_HAS_DIGIT = re.compile(r"\d")

# ── Location-section detection ───────────────────────────────────────────────────
# BOE environmental resolutions carry the place information under a heading like
# "Objeto, descripción y localización del proyecto." / "Localización del proyecto."
# / "Ubicación del proyecto.". Running NER over the WHOLE document floods the
# output with false positives (taxa lists, admin bodies, citations); restricting to
# this section is far cleaner. If no such heading exists (~22% of docs), we fall
# back to the whole document so nothing is lost.
_LOC_HEADING = re.compile(
    r"(?:descripci[oó]n\s+y\s+localizaci[oó]n"
    r"|localizaci[oó]n\s+y\s+descripci[oó]n"
    r"|localizaci[oó]n"
    r"|ubicaci[oó]n"
    r"|emplazamiento)"
    r"\s+del?\s+proyecto",
    re.IGNORECASE,
)
# Start of the NEXT section — where a location section ends. A heading-like line:
# a dash/number/ordinal bullet, or a known follow-on section title, at line start.
_NEXT_SECTION = re.compile(
    r"(?:^|\n)\s*(?:"
    r"[–—\-•·]\s+[A-ZÁÉÍÓÚ]"                       # "– Promotor…", "- Tramitación…"
    r"|\d{1,2}(?:\.\d{1,2})*[.\)]\s+[A-ZÁÉÍÓÚ]"    # "2. …", "3.1 …"
    r"|(?:Primero|Segundo|Tercero|Cuarto|Quinto|Sexto|S[eé]ptimo|Octavo|Noveno|D[eé]cimo)\b"
    r"|(?:Promotor|Tramitaci[oó]n|An[aá]lisis|Resumen|Alternativas?|"
    r"Caracterizaci[oó]n|Elementos\s+ambientales|Antecedentes|"
    r"Fundamentos\s+de\s+[Dd]erecho|Consultas|Resoluci[oó]n)\b"
    r")",
)
_SECTION_MAXLEN = 2600   # cap a section window so a missing boundary can't run away


def _classify(name_norm: str) -> str:
    head = name_norm.split()
    first = head[0] if head else ""
    for typ, heads in _TYPE_HINTS:
        if first in heads or (name_norm and any(h in head for h in heads)):
            return typ
    return "place"


def _is_admin(name_norm: str) -> bool:
    toks = set(name_norm.split())
    return bool(toks & _ADMIN_STOP)


@dataclass
class LocationFinding:
    name: str                       # canonical (GeoNames) name when verified, else raw
    type: str
    count: int
    mentions: List[dict]            # [{text, start, end}]
    verified: bool = False
    geonameid: Optional[str] = None
    province: Optional[str] = None
    region: Optional[str] = None
    match_type: Optional[str] = None   # exact | core | fuzzy (gazetteer), None if not
    confidence: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "verified": self.verified,
            "geonameid": self.geonameid,
            "province": self.province,
            "region": self.region,
            "match_type": self.match_type,
            "confidence": self.confidence,
            "count": self.count,
            "mentions": self.mentions,
        }


class LocationExtractor:
    # spaCy labels we consider location-bearing. ORG/MISC also carry protected-area
    # names ("Red Natura 2000") but are too noisy for v1; LOC only by default.
    _LABELS = {"LOC"}

    def __init__(
        self,
        model: str = "es_core_news_md",
        labels: Optional[set] = None,
        geo_db: Optional[str] = None,
        verify: bool = True,
    ):
        import spacy  # imported lazily so the species-only path pays nothing

        # Keep only what NER needs — much faster on long legal docs.
        self.nlp = spacy.load(
            model, disable=["parser", "lemmatizer", "morphologizer", "attribute_ruler"]
        )
        self.nlp.max_length = 2_000_000
        self.labels = labels or self._LABELS

        # Gazetteer verification backbone (optional; degrade gracefully if absent).
        self.resolver = None
        if verify and GeoNamesResolver is not None:
            try:
                self.resolver = GeoNamesResolver(geo_db)
            except Exception as e:  # index not built / unreadable
                print(f"[location_extractor] gazetteer disabled: {e}", flush=True)

    @staticmethod
    def _location_sections(text: str) -> List[tuple]:
        """Windows of text under a "localización/ubicación del proyecto" heading,
        each running to the next section boundary (capped). Returns [(abs_start,
        section_text), ...] merged; empty if the document has no such heading."""
        raw: List[list] = []
        for m in _LOC_HEADING.finditer(text):
            start = m.end()
            nb = _NEXT_SECTION.search(text, start)
            end = min(nb.start() if nb else len(text), start + _SECTION_MAXLEN)
            if end > start:
                raw.append([start, end])
        raw.sort()
        merged: List[list] = []
        for s, e in raw:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(s, text[s:e]) for s, e in merged]

    def extract(self, doc: dict, verified_only: bool = False) -> dict:
        text = doc.get("text") or ""
        sections = self._location_sections(text)
        # Restrict NER to the location section(s) when present (far less noise);
        # otherwise scan the whole document so nothing is lost.
        chunks = sections if sections else [(0, text)]
        scope = "section" if sections else "document"

        acc: Dict[str, LocationFinding] = {}
        for base, chunk in chunks:
            for ent in self.nlp(chunk).ents:
                if ent.label_ not in self.labels:
                    continue
                raw = ent.text.strip()
                norm = _strip_accents(raw)
                if not norm or _JUNK.match(norm) or _HAS_DIGIT.search(norm) or _is_admin(norm):
                    continue

                hit = self.resolver.resolve(raw) if self.resolver else None
                if hit is not None:
                    key = hit.geonameid
                    name, typ = hit.canonical, hit.type
                    verified, gid = True, hit.geonameid
                    province, region = hit.province, hit.region
                    match_type, confidence = hit.match_type, hit.score
                else:
                    if verified_only:
                        continue
                    key = "raw:" + norm
                    name, typ = raw, _classify(norm)
                    verified, gid = False, None
                    province = region = match_type = confidence = None

                f = acc.get(key)
                if f is None:
                    f = acc[key] = LocationFinding(
                        name=name, type=typ, count=0, mentions=[], verified=verified,
                        geonameid=gid, province=province, region=region,
                        match_type=match_type, confidence=confidence,
                    )
                f.count += 1
                if len(f.mentions) < 10:
                    f.mentions.append(
                        {"text": raw, "start": base + ent.start_char, "end": base + ent.end_char}
                    )

        # verified first, then by frequency, then name — a clean list up top.
        findings = sorted(acc.values(), key=lambda x: (not x.verified, -x.count, x.name))
        return {
            "_id": doc.get("_id"),
            "scope": scope,   # "section" = restricted to the localización section; "document" = whole-doc fallback
            "n_locations": len(findings),
            "n_verified": sum(1 for f in findings if f.verified),
            "locations": [f.as_dict() for f in findings],
        }
