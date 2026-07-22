"""
extractor.py — species extraction from a pre-ingested BOE-style document.

Input is a document that already carries `italicized_terms` (scientific names are
italicised in the source) plus the raw `text`. This is *not* a trained model: it
is candidate-generation + OTT verification, which is more reliable and explainable
than a net trained on ~421 weakly-labelled docs, and it fixes misspellings
("Accipiter gentiles" → "Accipiter gentilis") for free via the resolver.

Pipeline:
    1. candidates   italicized_terms  (∪ a conservative binomial scan of `text`)
    2. verify       resolve each through the local OTT index
                    - italics  → full cascade incl. fuzzy (curated, may be misspelt)
                    - text scan→ deterministic only (noisy source, no fuzzy)
    3. filter       keep only concepts at taxonomic ranks we care about
    4. dedupe       group by OTT id; keep provenance (raw variants, source, count)
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Vendored resolver (stdlib-only, no heavy deps) — ott_resolver.py, alongside this file.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from ott_resolver import OTTResolver  # noqa: E402

# Ranks we accept as "a species finding". Genus is kept (docs often name a genus
# only); anything above genus is dropped as too coarse to be a taxon mention.
_DEFAULT_ACCEPT_RANKS = {
    "species", "subspecies", "variety", "varietas", "form", "forma",
    "genus", "species group", "species subgroup", "infraspecificname",
}

# Conservative "Genus epithet" scan for species that are not italicised. Requires
# a capitalised genus (≥3 letters) + a lowercase epithet (≥3 letters). The OTT
# resolver is the real precision filter; this only proposes candidates.
_BINOMIAL = re.compile(r"\b([A-Z][a-zäöüïëç]{2,})\s+([a-zäöüïëç]{3,})\b")


@dataclass
class SpeciesFinding:
    canonical: str
    ott_id: str
    rank: str
    kingdom: Optional[str]
    match_type: str            # exact | synonym | abbrev | fuzzy
    score: float
    count: int                 # number of mentions that mapped here
    mentions: List[dict]       # [{text, start, end}] — char offsets into doc text
    sources: List[str]         # subset of {"italics", "text"}

    def as_dict(self) -> dict:
        return {
            "canonical": self.canonical,
            "ott_id": self.ott_id,
            "rank": self.rank,
            "kingdom": self.kingdom,
            "match_type": self.match_type,
            "score": self.score,
            "count": self.count,
            "mentions": self.mentions,
            "sources": self.sources,
        }


@dataclass
class _Acc:
    hit: object
    mentions: list = field(default_factory=list)   # [{text, start, end}]
    seen: set = field(default_factory=set)          # (text, start) dedupe keys
    sources: set = field(default_factory=set)


class SpeciesExtractor:
    def __init__(
        self,
        resolver: Optional[OTTResolver] = None,
        db_path: Optional[str] = None,
        scan_text: bool = True,
        accept_ranks: Optional[set] = None,
    ):
        self.resolver = resolver or OTTResolver(
            db_path or os.environ.get("SPECIES_QA_OTT_DB") or None
        )
        self.scan_text = scan_text
        self.accept_ranks = accept_ranks or _DEFAULT_ACCEPT_RANKS

    # -- candidate generation --------------------------------------------------

    @staticmethod
    def _find_spans(text: str, term: str, limit: int = 25) -> List[tuple]:
        """Char offsets of every (case-insensitive) occurrence of `term` in `text`."""
        spans: List[tuple] = []
        if not text or not term:
            return spans
        tl, q = text.lower(), term.lower()
        i = 0
        while len(spans) < limit:
            j = tl.find(q, i)
            if j < 0:
                break
            spans.append((j, j + len(term)))
            i = j + len(term)
        return spans

    def _candidates(self, doc: dict, scan_text: Optional[bool] = None) -> List[tuple]:
        """Return [(raw_string, source, allow_fuzzy, [(start,end), ...]), ...]."""
        scan = self.scan_text if scan_text is None else scan_text
        text = doc.get("text") or ""
        cands: List[tuple] = []
        seen = set()
        italics = doc.get("italicized_terms") or []
        for t in italics:
            t = (t or "").strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                # locate the italic term in the text so mentions get offsets
                cands.append((t, "italics", True, self._find_spans(text, t)))
        # Scan the raw text when explicitly asked, OR when the document has no
        # italics markup at all (e.g. plain {id,title,text} docs from the triage
        # GUI / SIBiLS FTP) — otherwise such docs would yield nothing.
        if scan or not italics:
            for m in _BINOMIAL.finditer(text):
                raw = m.group(0)
                if raw.lower() not in seen:
                    seen.add(raw.lower())
                    cands.append((raw, "text", False, [(m.start(), m.end())]))
        return cands

    # -- main ------------------------------------------------------------------

    def extract(self, doc: dict, scan_text: Optional[bool] = None) -> dict:
        """Extract species. `scan_text` overrides the instance default for this
        call only (None → use self.scan_text), so concurrent requests with
        different settings don't race on shared state."""
        acc: Dict[str, _Acc] = {}
        for raw, source, allow_fuzzy, spans in self._candidates(doc, scan_text):
            hit = self.resolver.resolve(raw, fuzzy=allow_fuzzy)
            if not hit:
                continue
            if self.accept_ranks and hit.rank and hit.rank.lower() not in self.accept_ranks:
                continue
            a = acc.get(hit.ott_id)
            if a is None:
                a = acc[hit.ott_id] = _Acc(hit=hit)
            a.sources.add(source)
            # one mention per located occurrence; fall back to an offset-less
            # mention when the raw string is not found verbatim in the text.
            for (s, e) in (spans or [(None, None)]):
                key = (raw, s)
                if key not in a.seen:
                    a.seen.add(key)
                    a.mentions.append({"text": raw, "start": s, "end": e})
            # prefer the strongest match_type seen for this concept
            if hit.score > a.hit.score:
                a.hit = hit

        findings = [
            SpeciesFinding(
                canonical=a.hit.canonical,
                ott_id=a.hit.ott_id,
                rank=a.hit.rank,
                kingdom=a.hit.kingdom,
                match_type=a.hit.match_type,
                score=a.hit.score,
                count=len(a.mentions),
                mentions=sorted(a.mentions, key=lambda m: (m["start"] is None, m["start"] or 0)),
                sources=sorted(a.sources),
            )
            for a in acc.values()
        ]
        # most-mentioned first, then alphabetical for stability
        findings.sort(key=lambda f: (-f.count, f.canonical))

        return {
            "_id": doc.get("_id"),
            "n_species": len(findings),
            "species": [f.as_dict() for f in findings],
        }
