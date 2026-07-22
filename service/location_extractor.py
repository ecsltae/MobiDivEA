"""
location_extractor.py — location extraction from a Spanish (BOE) document.

Off-the-shelf multilingual NER (spaCy `es_core_news_md`) + light rule cleanup.
There are NO gold location labels in the training data, so this is unsupervised:
spaCy proposes LOC/MISC entities, we drop administrative/legal noise it mislabels
as places, dedupe, and attach a coarse type (river / province / protected area /
water body / relig / trail / place) plus offsets for GUI highlighting.

This is deliberately simpler and noisier than the species side (which has the OTT
backbone to verify against). A Spanish GeoNames gazetteer could later verify and
normalize these the way OTT verifies species — see README roadmap.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional


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
    name: str
    type: str
    count: int
    mentions: List[dict]     # [{text, start, end}]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "count": self.count,
            "mentions": self.mentions,
        }


class LocationExtractor:
    # spaCy labels we consider location-bearing. ORG/MISC also carry protected-area
    # names ("Red Natura 2000") but are too noisy for v1; LOC only by default.
    _LABELS = {"LOC"}

    def __init__(self, model: str = "es_core_news_md", labels: Optional[set] = None):
        import spacy  # imported lazily so the species-only path pays nothing

        # Keep only what NER needs — much faster on long legal docs.
        self.nlp = spacy.load(
            model, disable=["parser", "lemmatizer", "morphologizer", "attribute_ruler"]
        )
        self.nlp.max_length = 2_000_000
        self.labels = labels or self._LABELS

    def extract(self, doc: dict) -> dict:
        text = doc.get("text") or ""
        acc: Dict[str, LocationFinding] = {}
        for ent in self.nlp(text).ents:
            if ent.label_ not in self.labels:
                continue
            raw = ent.text.strip()
            norm = _strip_accents(raw)
            if not norm or _JUNK.match(norm) or _HAS_DIGIT.search(norm) or _is_admin(norm):
                continue
            f = acc.get(norm)
            if f is None:
                f = acc[norm] = LocationFinding(
                    name=raw, type=_classify(norm), count=0, mentions=[]
                )
            f.count += 1
            if len(f.mentions) < 10:
                f.mentions.append({"text": raw, "start": ent.start_char, "end": ent.end_char})

        findings = sorted(acc.values(), key=lambda x: (-x.count, x.name))
        return {
            "_id": doc.get("_id"),
            "n_locations": len(findings),
            "locations": [f.as_dict() for f in findings],
        }
