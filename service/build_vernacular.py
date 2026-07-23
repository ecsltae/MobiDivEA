"""
build_vernacular.py — build the Spanish vernacular (common-name → scientific-name)
gazetteer from a Wikidata SPARQL dump.

Source (one-off, ~32k pairs, free):
    curl -s -G https://query.wikidata.org/sparql \
      --data-urlencode 'query=SELECT ?sci ?cn WHERE { ?i wdt:P225 ?sci; wdt:P1843 ?cn.
                        FILTER(LANG(?cn)="es") }' \
      -H 'Accept: text/tab-separated-values' -A 'app/1.0 (mail)' -o vernacular_raw.tsv

Common names collide hard with everyday Spanish ("papa"=potato, "alcalde"=mayor),
so this is precision-filtered at build time: keep multi-word names (distinctive:
"águila real", "sisón común") and long single words ("quebrantahuesos",
"avutarda"); drop short or common-word single tokens. Output is small enough to
ship in the repo. The extractor scans text for these, then resolves the paired
scientific name through OTT (same backbone as italic/text species).
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Dict, Set

# Build-time only (not a runtime dependency): Spanish word-frequency, used to drop
# common names that are really everyday words ("segundo", "relleno", "San Juan").
try:
    from wordfreq import zipf_frequency as _zipf
except Exception:  # pragma: no cover
    def _zipf(w: str, lang: str) -> float:  # frequency filter becomes a no-op
        return 0.0

# Zipf thresholds (higher = more common). A single-token common name more frequent
# than this is treated as an everyday word, not a species name; a multi-word name
# is dropped only if EVERY token is that common (kills "San Juan", keeps "milano
# real"). Tuned so distinctive fauna (milano 3.15, buitre 3.37, encina 3.17) survive
# while ordinals/construction terms (segundo 5.35, relleno 3.89, española 4.96) go.
_ZIPF_SINGLE = 3.6
_ZIPF_MULTI_ALL = 4.3

_OUT = Path(__file__).resolve().parent / "vernacular_es.tsv"

# Single-token common names that are also frequent Spanish words / too generic —
# never accept these as a species mention. (Multi-word names bypass this.)
_STOP: Set[str] = {
    "papa", "alcalde", "quimera", "gato", "rata", "raton", "mora", "lino",
    "pino", "haya", "cana", "junco", "sauce", "roble", "encina", "chopo",
    "alamo", "olmo", "fresno", "aliso", "brezo", "tomillo", "romero", "esparto",
    "carrizo", "aliaga", "retama", "jara", "lentisco", "madrono", "acebuche",
    "oso", "lobo", "zorro", "conejo", "liebre", "perdiz", "paloma", "gorrion",
    "estornino", "abeja", "avispa", "mosca", "mosquito", "hormiga", "trucha",
    "carpa", "barbo", "anguila", "sardina", "boga", "trigo", "cebada", "avena",
    "centeno", "vid", "olivo", "naranjo", "limonero", "almendro", "nogal",
    "castano", "abedul", "tejo", "sabina", "enebro", "durillo", "cornicabra",
    "calafate", "corcho", "grama", "cardo", "ortiga", "hinojo", "malva",
    # word-like common names that slip past the frequency filter (rare words that
    # are nonetheless not species references in this corpus)
    "carrete", "relleno", "carrera", "coladero", "regenerado",
    # harvested from a corpus scan: bare tokens that only ever resolved to the
    # WRONG taxon (a plant/fish/exotic) — their correct-species multi-word forms
    # ("aguilucho cenizo", "halcón peregrino") aren't in the gazetteer anyway, so
    # dropping the bare token loses nothing real.
    "cenizo", "paramo", "lucia", "merma", "boqueron", "morron", "podas",
    "cruceta", "canario", "serra", "innominado", "nerio", "peregrino", "puerco",
    "lupulo", "raya", "sargo", "aguja", "lija", "breca", "pardo", "blanca",
    "comun", "pinta", "rubio", "obispo", "fraile", "araña", "estrella",
    "pedrosa", "rellena", "plumero",
    # everyday words / adjectives that resolved to a taxon in the audit
    "acido", "voladora", "encaje", "terciaria", "trebolillo", "estepa",
    "babosa", "sabia", "lechera", "vieja", "espino",
}

# minimum length for a *single-token* common name to be kept (multi-word: always).
# The frequency filter now removes common words, so this can be short enough to
# keep distinctive short fauna names (sisón, lince, nutria, sepia).
_MIN_SINGLE = 5

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    return _WS.sub(" ", s).strip()


def build(raw_tsv: Path, out: Path = _OUT) -> None:
    index: Dict[str, dict] = {}   # cn_norm -> {"display":..., "sci": set()}
    kept = dropped = 0
    with open(raw_tsv, encoding="utf-8") as fh:
        next(fh, None)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            sci = parts[0].strip().strip('"')
            cn = parts[1].strip()
            cn = re.sub(r'"?@es$', "", cn).strip().strip('"')
            if not sci or not cn:
                continue
            cn_norm = _norm(cn)
            if not cn_norm or any(ch.isdigit() for ch in cn_norm):
                dropped += 1
                continue
            toks = cn_norm.split()
            # Frequency MUST be checked on the accent-preserved lowercase form —
            # wordfreq doesn't know accent-stripped words ("órgano" is common,
            # "organo" reads as rare), which would silently disable the filter.
            acc_toks = cn.lower().split()
            if len(toks) == 1:
                if cn_norm in _STOP or len(cn_norm) < _MIN_SINGLE:
                    dropped += 1
                    continue
                if _zipf(acc_toks[0] if acc_toks else cn_norm, "es") >= _ZIPF_SINGLE:
                    dropped += 1
                    continue
            else:
                # drop only if EVERY token is a common word (proper-noun / generic
                # phrase like "San Juan"); keep "milano real", "águila imperial"
                if acc_toks and all(_zipf(t, "es") >= _ZIPF_MULTI_ALL for t in acc_toks):
                    dropped += 1
                    continue
            # scientific must look like a binomial/uninomial (letters, ≥1 word)
            if not re.match(r"^[A-Z][a-zA-Z-]+(?:\s+[a-z][a-zA-Z-]+)*$", sci):
                dropped += 1
                continue
            # drop "common names" that are just the Latin genus token (e.g.
            # "Salvia"→Salvia officinalis, "Centaurea"→Centaurea): these match the
            # binomial in the text, not a real vernacular use.
            if len(toks) == 1 and toks[0] == _norm(sci.split()[0]):
                dropped += 1
                continue
            e = index.setdefault(cn_norm, {"display": cn, "sci": set()})
            e["sci"].add(sci)
            kept += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("common_norm\tcommon_display\tscientific\n")
        for cn_norm in sorted(index):
            e = index[cn_norm]
            fh.write(f"{cn_norm}\t{e['display']}\t{'|'.join(sorted(e['sci']))}\n")
    print(f"kept {kept} pairs → {len(index)} distinct common names, dropped {dropped}")
    print(f"→ {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", default=_OUT, type=Path)
    args = ap.parse_args()
    build(args.raw, args.out)
