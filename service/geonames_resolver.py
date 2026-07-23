"""
geonames_resolver.py — offline Spanish place-name → GeoNames id resolution.

The *verification* backbone for locations, mirroring `ott_resolver.py` for species:
NER (spaCy) proposes location-bearing spans (recognition); this resolves each span
to a canonical GeoNames concept (resolution), which:

  * filters NER noise — administrative bodies, legal refs, stray tokens ("Además")
    don't resolve to a geographic feature, so they drop out;
  * normalizes to a canonical name + a stable `geonameid`;
  * assigns an accurate `type` from the GeoNames feature code (river / water_body /
    relief / protected_area / province / region / municipality / …) instead of a
    head-noun guess;
  * attaches admin context (province / autonomous community).

Only *geographic* feature classes are indexed — A (admin), P (populated places),
H (hydrography), T (relief), L (parks/reserves), V (forest). Class S (buildings &
facilities: 19k hotels, museums, restaurants, …) and R (roads) are excluded on
purpose: they are the bulk of GeoNames but not the "where did the fieldwork
happen" locations this service answers.

Resolution cascade for one NER span:

    1. normalize     accents, case, punctuation, whitespace.
    2. exact(full)   the whole span, e.g. "Sierra de Guara" stored verbatim.
    3. head-strip    peel a leading Spanish geographic generic ("río", "provincia
                     de", "término municipal de", …) — which also yields a type
                     hint — and match the remaining core, e.g. "río Sosa" → "Sosa",
                     "provincia de Huesca" → "Huesca".
    4. fuzzy         conservative misspelling match (prefix-blocked), like OTT.

When a normalized name maps to several places (e.g. "Huesca" the city vs the
province), candidates are ranked: a head-noun type hint wins (provincia → ADM2),
otherwise by a feature-class priority and population, deterministically.

Build once (needs the GeoNames Spain dump ES.txt, ~3 MB, from
https://download.geonames.org/export/dump/ES.zip):

    python geonames_resolver.py build \
        --txt ../classifier/data/geonames/ES.txt \
        --db  ../classifier/data/processed/geonames_index.sqlite

Query:
    from geonames_resolver import GeoNamesResolver
    r = GeoNamesResolver()
    hit = r.resolve("río Sosa")     # -> GeoHit(geonameid=..., type='river', ...)
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import threading
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from rapidfuzz.fuzz import ratio as _rf_ratio  # type: ignore

    def _similarity(a: str, b: str) -> float:
        return _rf_ratio(a, b) / 100.0
except Exception:  # pragma: no cover
    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()


_DEFAULT_DB = (
    Path(__file__).resolve().parents[1]
    / "classifier/data/processed/geonames_index.sqlite"
)

# Feature classes we index. S (spots: hotels/museums/restaurants), R (roads),
# U (undersea) are excluded — see module docstring.
_KEEP_CLASSES = {"A", "P", "H", "T", "L", "V"}

# Fuzzy guards (place names are longer/safer than binomials, but stay conservative).
_FUZZY_MIN_SIM = 0.90
_FUZZY_MIN_LEN = 6
_FUZZY_MAX_CANDIDATES = 400

# ── Type mapping: (feature_class, feature_code) → our coarse type ────────────────
# Order within a class: most specific codes first.
_CODE_TYPE = {
    # admin
    "ADM1": "region",         # comunidad autónoma
    "ADM2": "province",       # provincia
    "ADM3": "comarca",
    "ADM4": "municipality",
    # hydrography
    "STM": "river", "STMI": "river", "STMS": "river", "STMM": "river",
    "CNL": "canal", "CNLI": "canal",
    "LK": "water_body", "LKS": "water_body", "RSV": "water_body",
    "PND": "water_body", "PNDS": "water_body", "LGN": "water_body",
    "MRSH": "wetland", "SWMP": "wetland",
    "SPNG": "spring", "WLL": "spring",
    # relief
    "MT": "relief", "MTS": "relief", "PK": "relief", "PKS": "relief",
    "HLL": "relief", "HLLS": "relief", "RDGE": "relief", "SRA": "relief",
    "CLF": "relief", "PLN": "relief", "VAL": "relief", "PASS": "pass",
    "CAPE": "cape", "PT": "cape",
    # protected / parks
    "PRK": "protected_area", "RESN": "protected_area", "RESV": "protected_area",
    "RESW": "protected_area", "RESF": "protected_area",
    # forest
    "FRST": "forest",
}
# Fallback by class when the code isn't in the table above.
_CLASS_TYPE = {"A": "admin", "P": "municipality", "H": "water_body",
               "T": "relief", "L": "protected_area", "V": "place"}
# Ranking priority when a bare name matches several places and there's no type hint.
# Higher = preferred. Populated places usually win in running text; then admin.
_CLASS_PRIORITY = {"P": 5, "A": 4, "T": 3, "H": 3, "L": 2, "V": 1}


def _type_of(fclass: str, fcode: str) -> str:
    return _CODE_TYPE.get(fcode) or _CLASS_TYPE.get(fclass, "place")


# ── Head-noun stripping: leading Spanish geographic generics → type hint ─────────
# Each generic maps to the feature CLASS(es) it implies, used to disambiguate.
_HEAD_GENERICS: List[Tuple[str, set]] = [
    (r"terminos? municipal(?:es)?", {"P", "A"}),
    (r"municipios?", {"P", "A"}),
    (r"localidad(?:es)?", {"P"}),
    (r"nucleos? de poblacion", {"P"}),
    (r"pedania", {"P"}),
    (r"provincias?", {"A"}),
    (r"comarcas?", {"A"}),
    (r"comunidad(?:es)? autonoma(?:s)?", {"A"}),
    (r"parques? (?:natural(?:es)?|nacional(?:es)?|regional(?:es)?)?", {"L"}),
    (r"reservas? (?:natural(?:es)?|de la biosfera)?", {"L"}),
    (r"parajes? natural(?:es)?", {"L"}),
    (r"espacios? natural(?:es)?", {"L"}),
    (r"monumentos? natural(?:es)?", {"L"}),
    (r"rios?", {"H"}), (r"arroyos?", {"H"}), (r"barrancos?", {"H"}),
    (r"ramblas?", {"H"}), (r"riberas?", {"H"}), (r"rieras?", {"H"}),
    (r"regatos?", {"H"}), (r"regueros?", {"H"}),
    (r"embalses?", {"H"}), (r"lagunas?", {"H"}), (r"pantanos?", {"H"}),
    (r"charcas?", {"H"}), (r"humedal(?:es)?", {"H"}), (r"balsas?", {"H"}),
    (r"marismas?", {"H"}), (r"estanques?", {"H"}),
    (r"sierras?", {"T"}), (r"montes?", {"T"}), (r"montanas?", {"T"}),
    (r"puertos?", {"T"}), (r"valles?", {"T"}), (r"picos?", {"T"}),
    (r"collados?", {"T"}), (r"cerros?", {"T"}), (r"lomas?", {"T"}),
    (r"altos?", {"T"}), (r"penas?", {"T"}), (r"macizos?", {"T"}),
    (r"cordilleras?", {"T"}), (r"cabos?", {"T"}), (r"puntas?", {"T"}),
]
# Connector words between the generic and the proper name: "de", "del", "de la"…
_CONN = r"(?:de\s+(?:la\s+|los\s+|las\s+|el\s+)?|del\s+)?"
_HEAD_RE = [
    (re.compile(rf"^{g}\s+{_CONN}", re.IGNORECASE), classes)
    for g, classes in _HEAD_GENERICS
]

_PUNCT = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")

# Single-token spans that are common Spanish words / cardinal directions which
# collide with tiny hamlet names in GeoNames — never a real location reference in
# this corpus ("al Oeste", "Ley de Montes", "Centro de transformación"). Rejected
# both as a bare span and as a head-stripped core. Multi-word names that merely
# contain these (e.g. "Laguna de la Nava") resolve via the exact full-name path
# before this ever applies.
_COMMON_STOP = {
    "norte", "sur", "este", "oeste", "noreste", "noroeste", "sureste",
    "suroeste", "nordeste", "sudeste", "sudoeste", "centro", "occidental",
    "oriental", "montes", "laguna", "pantano", "embalse", "arroyo", "sierra",
}


def normalize_name(name: str) -> str:
    """Accents off, lower-case, punctuation → space, whitespace collapsed."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def strip_head(name_norm: str) -> Tuple[str, Optional[set]]:
    """Peel a leading geographic generic. Returns (core, type_hint_classes|None)."""
    for rx, classes in _HEAD_RE:
        m = rx.match(name_norm)
        if m and m.end() < len(name_norm):
            core = name_norm[m.end():].strip()
            if core:
                return core, classes
    return name_norm, None


def _prefix3(name_norm: str) -> str:
    return name_norm.replace(" ", "")[:3]


# ── Query-side resolver ──────────────────────────────────────────────────────────

@dataclass
class GeoHit:
    geonameid: str          # "geonames:462523"
    canonical: str          # GeoNames preferred name
    type: str               # river | province | municipality | protected_area | …
    fclass: str             # raw GeoNames feature class (A/P/H/T/L/V)
    fcode: str              # raw GeoNames feature code (ADM2/STM/PPL/…)
    province: Optional[str]  # ADM2 name, when known
    region: Optional[str]   # ADM1 name (comunidad autónoma), when known
    match_type: str         # "exact" | "core" | "fuzzy"
    score: float
    query: str


class GeoNamesResolver:
    """Read-only resolver over the prebuilt SQLite gazetteer index."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or os.environ.get("SPECIES_QA_GEO_DB") or _DEFAULT_DB)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"GeoNames index not found at {self.db_path}. Build it once with:\n"
                f"    python geonames_resolver.py build --txt <ES.txt> "
                f"--db {self.db_path}"
            )
        self._local = threading.local()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(
            f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False
        )

    @property
    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = self._local.con = self._connect()
        return con

    # -- public API ------------------------------------------------------------

    def resolve(self, name: str, fuzzy: bool = False) -> Optional[GeoHit]:
        """Resolve one NER span → GeoHit, or None.

        Fuzzy matching is OFF by default: unlike species binomials, Spanish
        toponyms collide heavily with common words, verb forms, species common
        names and government terms, so a fuzzy pass produces mostly wrong hits
        (audited precision collapse). Exact + head-stripped-core only.
        """
        norm = normalize_name(name)
        if not norm:
            return None
        if " " not in norm and norm in _COMMON_STOP:
            return None
        # 1. whole span verbatim (e.g. "sierra de guara", "rio cinca") — the
        #    gazetteer name already carries the generic, so the type is right.
        hit = self._lookup(norm, None, "exact")
        if hit:
            return hit
        # 2. peel a leading generic and match the core, but ONLY to a feature of
        #    the hinted class — "río X" must resolve to hydrography, never to a
        #    same-named town (which would be the wrong entity AND the wrong type).
        core, hint = strip_head(norm)
        if core != norm and hint and not (" " not in core and core in _COMMON_STOP):
            hit = self._lookup(core, hint, "core", restrict=True)
            if hit:
                return hit
        # 3. fuzzy — opt-in only (see docstring).
        if fuzzy:
            return self._fuzzy(core if core != norm else norm, hint)
        return None

    # -- cascade helpers -------------------------------------------------------

    def _lookup(
        self, norm: str, hint: Optional[set], match_type: str, restrict: bool = False
    ) -> Optional[GeoHit]:
        ids = [
            r[0] for r in self._con.execute(
                "SELECT geonameid FROM names WHERE name_norm = ?", (norm,)
            ).fetchall()
        ]
        if not ids:
            return None
        rows = self._con.execute(
            f"SELECT geonameid, fclass, fcode, population FROM places "
            f"WHERE geonameid IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
        if restrict and hint:
            rows = [r for r in rows if r[1] in hint]
        if not rows:
            return None
        best = self._rank(rows, hint)
        return self._build_hit(best, norm, match_type, 1.0) if best else None

    def _fuzzy(self, norm: str, hint: Optional[set]) -> Optional[GeoHit]:
        if len(norm) < _FUZZY_MIN_LEN:
            return None
        cands = self._con.execute(
            "SELECT name_norm, geonameid FROM names WHERE prefix3 = ? LIMIT ?",
            (_prefix3(norm), _FUZZY_MAX_CANDIDATES),
        ).fetchall()
        best_sim, best_ids = 0.0, []
        by_name: Dict[str, list] = {}
        for cand_norm, gid in cands:
            by_name.setdefault(cand_norm, []).append(gid)
        for cand_norm, gids in by_name.items():
            sim = _similarity(norm, cand_norm)
            if sim > best_sim:
                best_sim, best_ids = sim, gids
        if best_ids and best_sim >= _FUZZY_MIN_SIM:
            rows = self._con.execute(
                f"SELECT geonameid, fclass, fcode, population FROM places "
                f"WHERE geonameid IN ({','.join('?' * len(best_ids))})",
                best_ids,
            ).fetchall()
            if hint:
                rows = [r for r in rows if r[1] in hint] or rows
            best = self._rank(rows, hint)
            return self._build_hit(best, norm, "fuzzy", round(best_sim, 3)) if best else None
        return None

    def _rank(self, rows: List[tuple], hint: Optional[set]) -> Optional[str]:
        """Pick the best place from (geonameid, fclass, fcode, population) rows:
        honour a class hint, else class priority + population."""
        if not rows:
            return None
        if len(rows) == 1:
            return rows[0][0]

        def key(r):
            gid, fclass, fcode, pop = r
            hint_match = 1 if (hint and fclass in hint) else 0
            # ADM2 should win "provincia de X" even against the same-named city.
            adm2_bonus = 1 if (hint and "A" in hint and fcode == "ADM2") else 0
            return (hint_match, adm2_bonus, _CLASS_PRIORITY.get(fclass, 0), pop or 0)

        return max(rows, key=key)[0]

    def _build_hit(self, gid: str, query: str, match_type: str, score: float) -> Optional[GeoHit]:
        row = self._con.execute(
            "SELECT name, fclass, fcode, province, region FROM places WHERE geonameid = ?",
            (gid,),
        ).fetchone()
        if not row:
            return None
        name, fclass, fcode, province, region = row
        return GeoHit(
            geonameid=f"geonames:{gid}",
            canonical=name,
            type=_type_of(fclass, fcode),
            fclass=fclass,
            fcode=fcode,
            province=province or None,
            region=region or None,
            match_type=match_type,
            score=score,
            query=query,
        )


# ── Index builder (one-off, offline) ─────────────────────────────────────────────

# GeoNames "geoname" table columns (tab-separated), see ES readme.txt.
_COL = {
    "geonameid": 0, "name": 1, "asciiname": 2, "alternatenames": 3,
    "fclass": 6, "fcode": 7, "admin1": 10, "admin2": 11, "population": 14,
}


def build_index(txt_path: Path, db_path: Path) -> None:
    """Parse the GeoNames Spain dump once into a queryable SQLite index."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    # Pass 1: read all kept rows into memory (small file), build admin-code → name.
    rows: List[list] = []
    adm1_name: Dict[str, str] = {}          # "51" -> "Aragón"
    adm2_name: Dict[Tuple[str, str], str] = {}  # ("51","22") -> "Huesca"
    with open(txt_path, "r", encoding="utf-8") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 15:
                continue
            fclass, fcode = p[_COL["fclass"]], p[_COL["fcode"]]
            if fclass not in _KEEP_CLASSES:
                continue
            rows.append(p)
            a1, a2 = p[_COL["admin1"]], p[_COL["admin2"]]
            if fcode == "ADM1" and a1:
                adm1_name[a1] = p[_COL["name"]]
            elif fcode == "ADM2" and a1 and a2:
                adm2_name[(a1, a2)] = p[_COL["name"]]

    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.executescript(
        """
        CREATE TABLE places (
            geonameid TEXT PRIMARY KEY, name TEXT, fclass TEXT, fcode TEXT,
            province TEXT, region TEXT, population INTEGER
        );
        CREATE TABLE names (name_norm TEXT, geonameid TEXT, is_alt INTEGER, prefix3 TEXT);
        """
    )

    place_rows, name_rows = [], []
    n_places = n_names = 0
    for p in rows:
        gid = p[_COL["geonameid"]]
        name = p[_COL["name"]]
        fclass, fcode = p[_COL["fclass"]], p[_COL["fcode"]]
        a1, a2 = p[_COL["admin1"]], p[_COL["admin2"]]
        try:
            pop = int(p[_COL["population"]] or 0)
        except ValueError:
            pop = 0
        region = adm1_name.get(a1)
        province = adm2_name.get((a1, a2))
        place_rows.append((gid, name, fclass, fcode, province, region, pop))
        n_places += 1

        # index the main name, asciiname, and every alternate name
        terms = {(name, 0), (p[_COL["asciiname"]], 0)}
        alts = p[_COL["alternatenames"]]
        if alts:
            for alt in alts.split(","):
                terms.add((alt, 1))
        for term, is_alt in terms:
            norm = normalize_name(term)
            if norm and len(norm) >= 2:
                name_rows.append((norm, gid, is_alt, _prefix3(norm)))
                n_names += 1

        if len(place_rows) >= 20_000:
            con.executemany("INSERT INTO places VALUES (?,?,?,?,?,?,?)", place_rows)
            con.executemany("INSERT INTO names VALUES (?,?,?,?)", name_rows)
            con.commit()
            place_rows, name_rows = [], []

    if place_rows:
        con.executemany("INSERT INTO places VALUES (?,?,?,?,?,?,?)", place_rows)
    if name_rows:
        con.executemany("INSERT INTO names VALUES (?,?,?,?)", name_rows)
    con.commit()

    con.execute("CREATE INDEX ix_names_norm ON names(name_norm)")
    con.execute("CREATE INDEX ix_names_prefix ON names(prefix3)")
    con.commit()
    con.close()
    print(f"Done: {n_places:,} places, {n_names:,} name keys → {db_path}", flush=True)


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the SQLite index from the GeoNames dump")
    b.add_argument("--txt", required=True, type=Path)
    b.add_argument("--db", default=_DEFAULT_DB, type=Path)

    q = sub.add_parser("resolve", help="resolve place names from the command line")
    q.add_argument("names", nargs="+")
    q.add_argument("--db", default=_DEFAULT_DB, type=Path)

    args = ap.parse_args()
    if args.cmd == "build":
        build_index(args.txt, args.db)
    else:
        r = GeoNamesResolver(args.db)
        for name in args.names:
            hit = r.resolve(name)
            if hit:
                print(
                    f"{name!r:40} → {hit.geonameid}  {hit.canonical!r} "
                    f"[{hit.type} · {hit.fclass}.{hit.fcode}] "
                    f"prov={hit.province} reg={hit.region} "
                    f"({hit.match_type} {hit.score})"
                )
            else:
                print(f"{name!r:40} → UNRESOLVED")


if __name__ == "__main__":
    _cli()
