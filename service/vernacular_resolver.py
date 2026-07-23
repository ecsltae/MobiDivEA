"""
vernacular_resolver.py — detect Spanish vernacular (common) species names in text.

Loads the `vernacular_es.tsv` gazetteer (common-name → scientific-name, built by
`build_vernacular.py` from Wikidata) and scans a document for known common names,
returning the scientific name(s) to resolve through OTT. This gives the species
extractor a third candidate source (alongside italics and the binomial text scan)
so documents that name taxa only in Spanish ("milano real", "avutarda") still
yield species.

Matching is exact, word-boundary, longest-phrase-first over accent-normalized
tokens, with original-text char offsets preserved for GUI highlighting. Common
names are ambiguous, so the gazetteer is already precision-filtered at build time
(multi-word or long single tokens only); callers should still treat these as
lower-confidence than binomials (the service tags them `source: vernacular`).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DEFAULT_TSV = Path(__file__).resolve().parent / "vernacular_es.tsv"
_WORD = re.compile(r"[0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
_MAX_NGRAM = 4


def _norm_token(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


class VernacularResolver:
    def __init__(self, tsv_path: Optional[Path] = None):
        self.tsv_path = Path(tsv_path or _DEFAULT_TSV)
        # common_norm -> (display, [scientific, ...])
        self.index: Dict[str, Tuple[str, List[str]]] = {}
        if not self.tsv_path.exists():
            raise FileNotFoundError(
                f"vernacular gazetteer not found at {self.tsv_path}. Build it with:\n"
                f"    python build_vernacular.py --raw vernacular_raw.tsv"
            )
        with open(self.tsv_path, encoding="utf-8") as fh:
            next(fh, None)  # header
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                cn_norm, display, sci = parts[0], parts[1], parts[2]
                self.index[cn_norm] = (display, sci.split("|"))
        # longest common name (in tokens) actually present, to bound the window
        self._max_ngram = min(
            _MAX_NGRAM, max((len(k.split()) for k in self.index), default=1)
        )

    def scan(self, text: str) -> List[dict]:
        """Return [{common, scientific:[...], start, end}] for every gazetteer hit,
        char offsets into `text`. Longest phrase wins; non-overlapping."""
        toks = [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(text)]
        norms = [_norm_token(t) for (t, _, _) in toks]
        out: List[dict] = []
        i, n = 0, len(toks)
        while i < n:
            hit = None
            for L in range(min(self._max_ngram, n - i), 0, -1):
                phrase = " ".join(norms[i:i + L])
                entry = self.index.get(phrase)
                if entry:
                    start, end = toks[i][1], toks[i + L - 1][2]
                    display = text[start:end]
                    # Reject Capitalised matches: genuine common-name mentions run
                    # lowercase in Spanish prose ("el milano real"), whereas a
                    # leading capital signals a proper noun — a place, person, org,
                    # or acronym (Burgo, Oso Pardo, SABIA) — or a Latin genus token
                    # ("Salvia"). This is the single biggest vernacular precision
                    # gain. (Sentence-initial common names are a small, rare loss.)
                    if display[:1].isupper():
                        break
                    hit = {
                        "common": display,
                        "scientific": entry[1],
                        "start": start,
                        "end": end,
                    }
                    i += L
                    break
            if hit:
                out.append(hit)
            else:
                i += 1
        return out


if __name__ == "__main__":
    import sys
    r = VernacularResolver()
    txt = " ".join(sys.argv[1:]) or (
        "Se observaron ejemplares de milano real y avutarda comun, "
        "asi como el aguila imperial iberica. La papa no cuenta."
    )
    for h in r.scan(txt):
        print(f"  {h['common']!r} @{h['start']}:{h['end']} → {h['scientific']}")
